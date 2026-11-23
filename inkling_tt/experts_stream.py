"""BFP8_B expert streaming for decode and block prefill on a single Blackhole."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

import torch
import ttnn

from . import dense, rowvm
from .device import has_default_l1_budget

_KERNELS = Path(__file__).resolve().parent / "kernels" / "experts"
_BF, _FP, _BFP8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b
_TILE, _RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
_L1, _DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
_EXPERTS, _TOPK, _BANKS = 128, 8, 8
_WIDTH, _HIDDEN = 2304, 1152
_GU_K_BLOCK = 6
_GU_LANES = 4
_GU_K, _DOWN_K = _WIDTH // 32, _HIDDEN // 32
_RING = 2
_VERIFY_RING = 2
_ACT_ROWS = 8
_DOWN_K_BLOCK = 12
_LOCK = Lock()


@dataclass(frozen=True)
class _Experts:
    gate_up: ttnn.Tensor
    down: ttnn.Tensor
    gate_tasks: tuple
    down_tasks: tuple
    fidelity: ttnn.MathFidelity


def _upload(device, packed):
    # Tile p = local_page * 8 + bank. Transposition moves whole 32x32
    # tiles, preserving the face-row groups used by the BFP8 converter.
    return ttnn.from_torch(
        packed.flatten(-2),
        device=device,
        dtype=_BFP8,
        layout=_TILE,
        memory_config=_DRAM,
    )


def _pack_gate_up(device, source, pairs):
    # Each bank owns two complete 64-channel groups and half of a
    # shared 32-channel group, split into alternating K blocks.
    full_group_tiles = _GU_K * 4
    bank_tiles = _GU_K * 9
    packed = torch.empty((_EXPERTS, bank_tiles * 32, _BANKS, 32), dtype=torch.bfloat16)

    def group_tiles(start, count):
        width = count * 32
        gate = source[:, start : start + width]
        up = source[:, _HIDDEN + start : _HIDDEN + start + width]
        block = torch.cat((gate, up), dim=1)
        block = block.reshape(_EXPERTS, 2 * count, 32, _GU_K // _GU_K_BLOCK, _GU_K_BLOCK, 32)
        return block.permute(0, 3, 4, 1, 5, 2)

    for bank in range(_BANKS):
        for group in range(2):
            offset = group * full_group_tiles * 32
            block = group_tiles(bank * 128 + group * 64, 2)
            packed[:, offset : offset + full_group_tiles * 32, bank, :] = block.reshape(_EXPERTS, -1, 32)
    for group, (first, second) in enumerate(pairs):
        blocks = group_tiles(1024 + group * 32, 1)
        for half, bank in enumerate((first, second)):
            packed[:, 2 * full_group_tiles * 32 :, bank, :] = blocks[:, half::2].reshape(_EXPERTS, -1, 32)
    return _upload(device, packed)


def _pack_down(device, source):
    columns = _WIDTH // 32 // _BANKS
    packed = torch.empty((_EXPERTS, columns, _DOWN_K * 32, _BANKS, 32), dtype=torch.bfloat16)
    tiles = source.reshape(_EXPERTS, columns, _BANKS, 32, _DOWN_K, 32)
    for bank in range(_BANKS):
        packed[..., bank, :] = tiles[:, :, bank].permute(0, 1, 3, 4, 2).reshape(_EXPERTS, columns, _DOWN_K * 32, 32)
    return _upload(device, packed)


def _workers(device):
    bounds = device.compute_with_storage_grid_size()
    assignment = device.get_optimal_dram_bank_to_logical_worker_assignment(
        ttnn.NOC.NOC_0,
        ttnn.MeshCoordinate(0, 0),
    )
    preferred = [assignment[bank] for bank in sorted(assignment)]
    local_readers = 2 * _GU_LANES
    shared_readers = _GU_LANES // 2
    down_readers = _WIDTH // 32 // _BANKS
    per_bank = max(local_readers + shared_readers, down_readers)
    if len(preferred) != _BANKS or bounds.x * bounds.y < _BANKS * per_bank:
        raise ValueError(f"expert streaming requires eight DRAM banks and at least {_BANKS * per_bank} workers")
    bank_order = sorted(range(_BANKS), key=lambda b: (preferred[b].x, preferred[b].y))
    pairs = tuple(zip(bank_order[::2], bank_order[1::2]))
    available = {(x, y) for y in range(bounds.y) for x in range(bounds.x)}
    workers = [[] for _ in preferred]
    # Favor workers in the forward NoC0 direction from the bank.
    for _ in range(per_bank):
        for near, group in zip(preferred, workers):

            def distance(c, near=near):
                dx = (c[0] - near.x) % bounds.x
                return (dx + 2 * abs(c[1] - near.y), c)

            worker = min(available, key=distance)
            group.append(worker)
            available.remove(worker)
    gate_local = []
    gate_shared = []
    for bank, group in enumerate(workers):
        for lane in range(_GU_LANES):
            for col in range(2):
                x, y = group[2 * lane + col]
                gate_local.append((x, y, bank * 4 + col * 2, bank, bank, lane, (x + y) % 4, col))
    shared = slice(local_readers, local_readers + shared_readers)
    for group, (first, second) in enumerate(pairs):
        cores = workers[first][shared] + workers[second][shared]
        for lane, (x, y) in enumerate(cores):
            gate_shared.append((x, y, 32 + group, first, second, lane, (x + y) % 4, 0))
    down = []
    for bank, group in enumerate(workers):
        for col, (x, y) in enumerate(group[:down_readers]):
            down.append((x, y, bank, col, (x + y) % 4))
    return (tuple(gate_local), tuple(gate_shared)), tuple(down), pairs


def prepare(device, gate_up_bf16, down_bf16):
    """Pack CPU BF16 HF Linear weights into BFP8_B DRAM. Call before tracing."""
    for weight, shape in (
        (gate_up_bf16, (_EXPERTS, _WIDTH, _WIDTH)),
        (down_bf16, (_EXPERTS, _WIDTH, _HIDDEN)),
    ):
        if tuple(weight.shape) != shape or weight.dtype != torch.bfloat16 or weight.device.type != "cpu":
            raise ValueError(f"expected a CPU BF16 weight tensor of shape {shape}")
    if device.arch() != ttnn.Arch.BLACKHOLE or (device.shape[0], device.shape[1]) != (1, 1):
        raise ValueError("expert streaming requires a single Blackhole device")
    gate_tasks, down_tasks, pairs = _workers(device)
    gu = _pack_gate_up(device, gate_up_bf16.detach(), pairs)
    down = _pack_down(device, down_bf16.detach())
    return _Experts(gu, down, gate_tasks, down_tasks, ttnn.MathFidelity.HiFi4)


def _cb(grid, index, fmt, page, pages=1, tile=None):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page, tile=tile)],
    )


@lru_cache(maxsize=64)
def _program(gate, tasks, fidelity, indices_spec, x_spec, out_spec, gamma_spec):
    groups = zip(tasks, (4, 2)) if gate else ((tasks, 1),)
    k = _GU_K if gate else _DOWN_K
    block_k = _GU_K_BLOCK if gate else _DOWN_K_BLOCK
    cbs, kernels = [], []
    for group, columns in groups:
        rt = [(ttnn.CoreCoord(x, y), list(args)) for x, y, *args in group]
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(rt[0][0], rt[0][0])])
        for core, _ in rt[1:]:
            grid = grid.merge(ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]))
        definitions = [
            ("EXPERTS_GATE", str(int(gate))),
            ("EXPERTS_ACT_ROWS", str(_ACT_ROWS)),
            ("EXPERTS_GU_LANES", str(_GU_LANES)),
        ]
        cbs.extend(
            [
                _cb(
                    grid, 0, _BF, 64 * _ACT_ROWS, k if gate else 2 * k, ttnn.TileDescriptor(ttnn.Tile([_ACT_ROWS, 32]))
                ),
                _cb(grid, 1, _BFP8, 1088, _RING * block_k * columns),
                _cb(grid, 2, _FP, 64),
                _cb(grid, 3, _FP, 64),
                _cb(grid, 16, _BF, 2048, (columns // 2 if gate else columns) * 2),
            ]
        )
        compile_args = [k, block_k, columns, _RING]
        kernels.extend(
            ttnn.KernelDescriptor(
                kernel_source=str(_KERNELS / source),
                core_ranges=grid,
                compile_time_args=compile_args + list(spec),
                runtime_args=rt,
                defines=definitions,
                config=config,
            )
            for source, spec, config in (
                ("weights.cpp", indices_spec, ttnn.ReaderConfigDescriptor()),
                ("rows.cpp", x_spec + out_spec + gamma_spec, ttnn.WriterConfigDescriptor()),
                (
                    "compute.cpp",
                    (),
                    ttnn.ComputeConfigDescriptor(
                        math_fidelity=fidelity,
                        math_approx_mode=False,
                        fp32_dest_acc_en=True,
                        dst_full_sync_en=gate,
                    ),
                ),
            )
        )
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernels


def _accessor(tensor):
    return tuple(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())


def _run(gate, x, ids, weight, gamma, output, tasks, fidelity):
    with _LOCK:
        desc, kernels = _program(
            gate,
            tasks,
            fidelity,
            _accessor(ids),
            _accessor(x),
            _accessor(output),
            () if gate else _accessor(gamma),
        )
        inputs = [x, ids, weight]
        reader_args = [ids.buffer_address(), weight.buffer_address()]
        writer_args = [x.buffer_address(), output.buffer_address()]
        if not gate:
            inputs.append(gamma)
            writer_args.append(gamma.buffer_address())
        for reader, writer in zip(kernels[::3], kernels[1::3]):
            reader.common_runtime_args = reader_args
            writer.common_runtime_args = writer_args
        desc.kernels = kernels
        ttnn.generic_op([*inputs, output], desc)
    return output


@lru_cache(maxsize=16)
def _stage_program(grid_width, specs):
    cbs, kernels = [], []
    first_worker = 0
    for kind, src_spec, dst_spec in specs:
        workers, pages, page_bytes, live_bytes, two_faces = (
            (24, _GU_K, 2048, 32, True),
            (1, 1, 16, 16, False),
            (1, 1, 4096, 32, False),
        )[kind]
        cores = [ttnn.CoreCoord(i % grid_width, i // grid_width) for i in range(first_worker, first_worker + workers)]
        first_worker += workers
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        cbs.append(_cb(grid, 0, _FP, 128))
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(_KERNELS / "stage.cpp"),
                core_ranges=grid,
                compile_time_args=[workers, pages, page_bytes, live_bytes, int(two_faces), *src_spec, *dst_spec],
                runtime_args=[(c, [task]) for task, c in enumerate(cores)],
                config=ttnn.ReaderConfigDescriptor(),
            )
        )
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernels


def _stage_dram(device, tensors):
    staged = list(tensors)
    kinds = [i for i, tensor in enumerate(tensors) if tensor.memory_config() == _DRAM]
    if not kinds:
        return tensors
    for i in kinds:
        tensor = tensors[i]
        staged[i] = ttnn.allocate_tensor_on_device(ttnn.Shape(tensor.shape), tensor.dtype, tensor.layout, device, _L1)
    specs = tuple((i, _accessor(tensors[i]), _accessor(staged[i])) for i in kinds)
    with _LOCK:
        desc, kernels = _stage_program(device.compute_with_storage_grid_size().x, specs)
        for i, kernel in zip(kinds, kernels):
            kernel.common_runtime_args = [tensors[i].buffer_address(), staged[i].buffer_address()]
        desc.kernels = kernels
        ttnn.generic_op([tensors[i] for i in kinds] + [staged[i] for i in reversed(kinds)], desc)
    return staged


def moe_decode(x, ids16, gamma, handle):
    """Return the BF16 [1,1,1,2304] branch from eight device-selected experts.

    IDs must be in [0,128) and correspond to the router's FP32 gamma slots.
    x and gamma use 32x32 TILE layout, and IDs use row-major layout.
    All inputs must be allocated interleaved tensors on the handle's device.
    Warm up before TTNN trace capture as with other generic operations.
    """
    rowvm._validate(x, (1, 1, 1, _WIDTH))
    rowvm._validate(gamma, (1, 1, 1, _TOPK))
    if x.dtype != _BF or gamma.dtype != _FP:
        raise ValueError("expert streaming requires BF16 x and FP32 gamma")
    if (
        tuple(ids16.shape) != (1, 1, 1, _TOPK)
        or ids16.dtype != ttnn.uint16
        or ids16.layout != _RM
        or ids16.is_sharded()
        or not ids16.is_allocated()
    ):
        raise ValueError("expected interleaved uint16 [1,1,1,8] row-major IDs")
    device = x.device()
    if any(t.device() != device for t in (ids16, gamma, handle.gate_up, handle.down)):
        raise ValueError("expert streaming tensors must be on the same device")
    x, ids16, gamma = _stage_dram(device, (x, ids16, gamma))
    hidden = ttnn.allocate_tensor_on_device(ttnn.Shape((1, _TOPK, 1, _HIDDEN)), _BF, _TILE, device, _L1)
    output = ttnn.allocate_tensor_on_device(ttnn.Shape(x.shape), _BF, _TILE, device, _L1)
    _run(True, x, ids16, handle.gate_up, gamma, hidden, handle.gate_tasks, handle.fidelity)
    return _run(False, hidden, ids16, handle.down, gamma, output, handle.down_tasks, handle.fidelity)


# Compact scratch tiles have two faces of eight rows. The existing expert
# pack still uses standard 32x32 BFP8 tiles.
_PREFILL_ROWS = 8
_PREFILL_KERNELS = _KERNELS.parent / "prefill"


def _prefill_capacity(tokens, rows):
    return (tokens * _TOPK + _EXPERTS * (rows - 1)) // rows


def _prefill_metadata(ids, gamma, tokens, rows=_PREFILL_ROWS):
    """Build forward and inverse token maps, with zero reserved for padding."""
    if (
        ids.device.type != "cpu"
        or gamma.device.type != "cpu"
        or ids.ndim != 2
        or ids.shape != gamma.shape
        or ids.shape[1] != _TOPK
        or ids.shape[0] > tokens
        or gamma.dtype != torch.float32
        or ids.dtype not in (torch.int16, torch.uint16, torch.int32, torch.int64)
    ):
        raise ValueError("routing requires CPU integer IDs and FP32 gamma of shape [valid_tokens, 8]")
    ids = ids.long()
    ids, slots = ids.sort(dim=-1)
    if ((ids < 0) | (ids >= _EXPERTS)).any() or (ids[:, 1:] == ids[:, :-1]).any():
        raise ValueError("each token must select eight distinct experts in [0,128)")
    flat_ids = ids.flatten()
    counts = flat_ids.bincount(minlength=_EXPERTS)
    blocks = (counts + rows - 1) // rows
    ends = blocks.cumsum(0)
    by_expert = flat_ids.argsort(stable=True)
    experts = flat_ids[by_expert]
    row = (ends - blocks)[experts] * rows
    row += torch.arange(by_expert.numel()) - (counts.cumsum(0) - counts)[experts]

    # At most rows - 1 padding rows per expert. Tensor shapes depend only on
    # the chunk size, so changing routes never compiles a new program.
    capacity = _prefill_capacity(tokens, rows)
    gather = torch.zeros(capacity * rows, dtype=torch.int32)
    gather[row] = (by_expert // _TOPK + 1).int()
    inverse = torch.zeros((tokens, _TOPK), dtype=torch.int32)
    inverse.flatten()[by_expert] = (row + 1).int()
    scores = torch.zeros((tokens, _TOPK), dtype=torch.float32)
    scores[: ids.shape[0]] = gamma.gather(-1, slots)
    scores = scores.reshape(-1, 32, _TOPK).transpose(1, 2)[..., None].expand(-1, -1, -1, 32)
    return (
        ends.int().reshape(1, 1, 1, _EXPERTS),
        gather.reshape(1, 1, 1, -1),
        inverse.reshape(1, 1, tokens // 32, 32 * _TOPK),
        scores.reshape(1, tokens // 32, _TOPK * 32, 32).contiguous(),
    )


def _prefill_alloc(device, shape, dtype, layout, memory=_L1):
    return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, layout, device, memory)


@dataclass(frozen=True)
class _PrefillWorkspace:
    tokens: int
    rows: int
    ends: ttnn.Tensor
    gather: ttnn.Tensor
    inverse: ttnn.Tensor
    gamma: ttnn.Tensor
    wide: ttnn.Tensor
    wide_tail: ttnn.Tensor | None
    hidden: ttnn.Tensor
    output: ttnn.Tensor
    ids: ttnn.Tensor
    broadcast_down: bool
    lanes: int = 0
    down_l1_blocks: int = 0
    weight_buffers: int = 2


def prepare_prefill(
    device,
    tokens,
    *,
    broadcast_down=False,
    full_tiles=False,
    lanes=0,
    wide_l1_blocks=None,
    weight_buffers=2,
    verify_rows=None,
):
    """Allocate reusable scratch for one prefill chunk on a single device.

    Use a separate workspace for each simultaneously live output. To trace the
    device programs, stage routing and run the full path eagerly with these
    buffers before capture. Stage new routing at the same addresses before replay.
    """
    if tokens < 32 or tokens > 2048 or tokens % 32:
        raise ValueError("prefill supports 32 to 2048 tokens in multiples of 32")
    if lanes and not full_tiles:
        raise ValueError("lane projections require full activation tiles")
    if weight_buffers not in (1, 2):
        raise ValueError("prefill projections reserve one or two expert weight buffers")
    max_lanes = prefill_lane_capacity(device)
    if not 0 <= lanes <= max_lanes:
        raise ValueError(f"prefill expert lanes must be between 0 and {max_lanes} for this worker grid, got {lanes}")
    rows = 32 if full_tiles else _PREFILL_ROWS
    page = rows * 64
    capacity = _prefill_capacity(tokens, rows)
    if verify_rows is not None:
        if tokens != 32 or full_tiles or not 2 <= verify_rows <= _PREFILL_ROWS:
            raise ValueError("expert verify uses one compact tile and two to eight live rows")
        capacity = verify_rows * _TOPK
    # Leave room for the native BF16 head's circular buffers during decode warmup.
    wide_blocks = min(capacity, 512) if full_tiles else capacity
    if lanes and wide_l1_blocks is None:
        # Lane projections keep two experts and two blocks resident per worker.
        # Rare padding blocks beyond the typical route spill to DRAM, and
        # chunks above 1,024 rows keep the whole compact buffer in DRAM.
        wide_l1_blocks = tokens // 4 + 32 if tokens <= 1024 else 1
    if wide_l1_blocks is not None:
        wide_blocks = min(capacity, wide_l1_blocks)
    down_l1_blocks = 0
    if lanes and tokens > 1024 and wide_blocks < capacity:
        # Keep space per worker for the lane CBs and both resident buckets.
        # Interleaving the same number of blocks across fewer L1 banks costs
        # more bytes per worker; the existing DRAM tail holds the remainder.
        grid = device.compute_with_storage_grid_size()
        workers = grid.x * grid.y
        default_blocks = 256 if weight_buffers == 1 else 600 if workers >= 130 else 540 if workers >= 120 else 480
        if not has_default_l1_budget(device):
            default_blocks -= 56
        down_l1_blocks = min(capacity, int(os.environ.get("INK_PREFILL_EXPERT_DOUT_L1", default_blocks)))
    routing_mem = _DRAM if weight_buffers == 1 else _L1
    wide_tail = (
        _prefill_alloc(device, (1, 1, (capacity - wide_blocks) * _GU_K, page // 2), _BF, _RM, _DRAM)
        if wide_blocks < capacity
        else None
    )
    return _PrefillWorkspace(
        tokens,
        rows,
        _prefill_alloc(device, (1, 1, 1, _EXPERTS), ttnn.uint32, _RM),
        _prefill_alloc(device, (1, 1, 1, capacity * rows), ttnn.uint32, _RM),
        _prefill_alloc(device, (1, 1, tokens // 32, 32 * _TOPK), ttnn.uint32, _RM),
        _prefill_alloc(device, (1, tokens // 32, _TOPK * 32, 32), _FP, _TILE, routing_mem),
        _prefill_alloc(device, (1, 1, wide_blocks * _GU_K, page // 2), _BF, _RM),
        wide_tail,
        _prefill_alloc(
            device,
            (1, 1, capacity * _DOWN_K, page // 2),
            _BF,
            _RM,
            _DRAM if (full_tiles and tokens >= 1024) or tokens > 1024 else _L1,
        ),
        _prefill_alloc(device, (1, 1, tokens, _WIDTH), _BF, _TILE, _DRAM),
        _prefill_alloc(device, (1, 1, tokens // 32, 32 * _TOPK), ttnn.uint32, _RM),
        broadcast_down,
        lanes,
        down_l1_blocks,
        weight_buffers,
    )


def stage_prefill_routing(routing, workspace):
    """Copy CPU (ids, gamma) into a preallocated workspace outside capture."""
    metadata = _prefill_metadata(*routing, workspace.tokens, workspace.rows)
    for host, target in zip(metadata, (workspace.ends, workspace.gather, workspace.inverse, workspace.gamma)):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=target.dtype, layout=target.layout), target)


def _prefill_work(width, height, count):
    workers = min(width * height, count)
    return [(i % width, i // width, i * count // workers, (i + 1) * count // workers) for i in range(workers)]


@lru_cache(maxsize=16)
def _prefill_route_program(select, tasks, specs, token_tiles, gather_bytes, rows):
    rt = [(ttnn.CoreCoord(x, y), [start, end]) for x, y, start, end in tasks]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, _ in rt])
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS.parent / "prefill" / "route.cpp"),
        core_ranges=grid,
        compile_time_args=[token_tiles, gather_bytes, rows, *sum(specs, ())],
        runtime_args=rt,
        defines=[("PREFILL_SELECT", str(int(select)))],
        config=ttnn.ReaderConfigDescriptor(),
    )
    map_bytes = 32 * _TOPK * 4
    scratch, data = (map_bytes, 16 * 4096) if select else (gather_bytes + 512, token_tiles * map_bytes)
    desc = ttnn.ProgramDescriptor(
        kernels=[kernel],
        cbs=[_cb(grid, 0, ttnn.uint32, scratch), _cb(grid, 1, ttnn.uint32, data)],
    )
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernel


def route_prefill(logits, bias, chunk_meta, workspace):
    """Write staged maps from FP32 router logits and the device's valid-row count.

    Correction bias participates in selection only. Exact ties select the
    lower expert ID. Each token's combine slots are ordered by expert ID.
    """
    scores = ttnn.sigmoid(logits)
    ranked = ttnn.add(scores, bias)
    bounds = logits.device().compute_with_storage_grid_size()
    token_tiles = workspace.tokens // 32
    gather_bytes = workspace.gather.shape[-1] * 4
    phases = (
        (
            True,
            tuple(_prefill_work(bounds.x, bounds.y, token_tiles)),
            (scores, ranked, chunk_meta, workspace.ids, workspace.gamma),
        ),
        (False, ((0, 0, 0, 1),), (workspace.ids, workspace.ends, workspace.gather, workspace.inverse)),
    )
    with _LOCK:
        for select, tasks, tensors in phases:
            desc, kernel = _prefill_route_program(
                select,
                tasks,
                tuple(_accessor(t) for t in tensors),
                token_tiles,
                gather_bytes,
                workspace.rows,
            )
            kernel.common_runtime_args = [t.buffer_address() for t in tensors]
            desc.kernels = [kernel]
            ttnn.generic_op(list(tensors), desc)


@lru_cache(maxsize=32)
def _prefill_io_program(combine, tasks, specs, map_bytes, token_tiles, rows, wide_blocks):
    rt = [(ttnn.CoreCoord(x, y), [start, end]) for x, y, start, end in tasks]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, _ in rt])
    args = [map_bytes, token_tiles, *sum(specs, ())]
    cbs = [_cb(grid, 2, ttnn.uint32, map_bytes)]
    wide_def = [("EXPERTS_WIDE_L1_BLOCKS", str(wide_blocks))]
    name = "combine_io" if combine else "gather"
    io_path = _PREFILL_KERNELS / f"expert_{name}.cpp" if rows == 32 else _KERNELS / f"prefill_{name}.cpp"
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(io_path),
            core_ranges=grid,
            compile_time_args=args,
            runtime_args=rt,
            defines=wide_def,
            config=ttnn.ReaderConfigDescriptor(),
        )
    ]
    if combine:
        cbs += [_cb(grid, 0, _BF, 2048, 4), _cb(grid, 1, _FP, 4096, _TOPK), _cb(grid, 16, _BF, 2048, 2)]
        if wide_blocks:
            cbs.append(_cb(grid, 3, _BF, 4096 + 64))
        compute = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        unpack = [ttnn.UnpackToDestMode.Default] * 64
        unpack[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute.unpack_to_dest_mode = unpack
        for source, config, args in (
            (_KERNELS / "prefill_combine.cpp", compute, []),
            (io_path, ttnn.WriterConfigDescriptor(), list(specs[-1])),
        ):
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(source),
                    core_ranges=grid,
                    compile_time_args=args,
                    runtime_args=rt,
                    defines=[("PREFILL_WRITE", "1")],
                    config=config,
                )
            )
    else:
        cbs += [_cb(grid, 0, _BF, rows * 64, 2), _cb(grid, 3, _BF, 2048, token_tiles + 1)]
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernels


@lru_cache(maxsize=32)
def _prefill_projection_program(
    gate, tasks, fidelity, specs, broadcast_down, rows, single_weights, wide_blocks, verify
):
    groups = zip(tasks, (4, 2)) if gate else ((tasks, 1),)
    k = _GU_K if gate else _DOWN_K
    block_k = _GU_K_BLOCK if gate else _DOWN_K_BLOCK
    cbs, kernels, semaphores = [], [], []
    for group, cols in groups:
        rt = [(ttnn.CoreCoord(x, y), list(args)) for x, y, *args in group]
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, _ in rt])
        if broadcast_down:
            semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=int(i == 2)) for i in range(3)]
        cbs.extend(
            [
                _cb(
                    grid,
                    0,
                    _BF,
                    rows * 64,
                    (1 if rows == 32 and gate else 2) * k,
                    ttnn.TileDescriptor(ttnn.Tile([rows, 32])),
                ),
                _cb(
                    grid,
                    1,
                    _BFP8,
                    1088,
                    (_VERIFY_RING * block_k if verify == 2 else (1 if gate and single_weights else 2) * k) * cols,
                ),
                _cb(grid, 2, ttnn.uint32, _EXPERTS * 4),
                _cb(grid, 3, ttnn.uint32, _EXPERTS * 4),
                _cb(grid, 16, _BF, 2048, 2 * (cols // 2 if gate else cols)),
            ]
        )
        definitions = [
            ("EXPERTS_GATE", str(int(gate))),
            ("EXPERTS_GU_LANES", str(_GU_LANES)),
            ("EXPERTS_BROADCAST", str(int(broadcast_down))),
            ("EXPERTS_WIDE_L1_BLOCKS", str(wide_blocks)),
        ]
        if verify:
            definitions.append(("EXPERTS_VERIFY", "1"))
        if verify == 2:
            definitions.extend((("EXPERTS_VERIFY_STREAM", "1"), ("EXPERTS_VERIFY_RING", str(_VERIFY_RING))))
        for source, accessor, config in (
            ("prefill_weights.cpp", specs[0], ttnn.ReaderConfigDescriptor()),
            ("prefill_rows.cpp", sum(specs, ()), ttnn.WriterConfigDescriptor()),
            (
                "prefill_matmul.cpp",
                (),
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=fidelity,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=gate,
                ),
            ),
        ):
            path = _KERNELS / source
            if verify == 2 and source == "prefill_weights.cpp":
                path = _KERNELS / "verify_weights.cpp"
            if rows == 32 and source != "prefill_weights.cpp":
                path = _PREFILL_KERNELS / source.replace("prefill_", "expert_")
            elif broadcast_down and source == "prefill_rows.cpp":
                path = _PREFILL_KERNELS / "expert_broadcast.cpp"
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(path),
                    core_ranges=grid,
                    compile_time_args=[k, block_k, cols, *accessor],
                    runtime_args=rt,
                    defines=definitions,
                    config=config,
                )
            )
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernels


def _compact(workspace, override=None):
    """Return the compact block buffer, its DRAM tail and the number of blocks before the tail."""
    buffer = workspace.wide if override is None else override
    tail = (workspace.wide_tail,) if workspace.wide_tail is not None else ()
    return buffer, tail, buffer.shape[-2] // _GU_K if tail else 0


def _prefill_io(x, workspace, combine, compact=None):
    bounds = x.device().compute_with_storage_grid_size()
    buffer, tail, wide_blocks = _compact(workspace, compact)
    output = workspace.output if combine else buffer
    if combine:
        inputs = (workspace.inverse, buffer, workspace.gamma)
        map_bytes = 32 * _TOPK * 4
        count = workspace.tokens // 32 * _GU_K
    else:
        inputs = (workspace.gather, workspace.ends, x)
        map_bytes = workspace.gather.shape[-1] * 4
        count = _GU_K
    with _LOCK:
        desc, kernels = _prefill_io_program(
            combine,
            tuple(_prefill_work(bounds.x, bounds.y, count)),
            tuple(_accessor(t) for t in (*inputs, *tail, output)),
            map_bytes,
            workspace.tokens // 32,
            workspace.rows,
            wide_blocks,
        )
        kernels[0].common_runtime_args = [t.buffer_address() for t in (*inputs, *tail, output)]
        if combine:
            kernels[-1].common_runtime_args = [output.buffer_address()]
        desc.kernels = kernels
        ttnn.generic_op([*inputs, *tail, output], desc)


def _prefill_project(gate, workspace, handle, *, verify=False):
    x, output = (workspace.wide, workspace.hidden) if gate else (workspace.hidden, workspace.wide)
    weights, tasks = (handle.gate_up, handle.gate_tasks) if gate else (handle.down, handle.down_tasks)
    broadcast_down = not gate and workspace.broadcast_down
    if broadcast_down:
        tasks = dense._prefill_multicast(x.device(), tasks)
    ends = workspace.ends
    tail = (workspace.wide_tail,) if workspace.wide_tail is not None else ()
    wide_blocks = workspace.wide.shape[-2] // _GU_K if tail else 0
    with _LOCK:
        desc, kernels = _prefill_projection_program(
            gate,
            tasks,
            handle.fidelity,
            tuple(_accessor(t) for t in (ends, x, output, *tail)),
            broadcast_down,
            workspace.rows,
            workspace.tokens > 1024 or workspace.weight_buffers == 1,
            wide_blocks,
            verify,
        )
        for reader, writer in zip(kernels[::3], kernels[1::3]):
            reader.common_runtime_args = [ends.buffer_address(), weights.buffer_address()]
            writer.common_runtime_args = [t.buffer_address() for t in (ends, x, output, *tail)]
        desc.kernels = kernels
        ttnn.generic_op([ends, x, weights, output, *tail], desc)


_LANE_GROUPS = 18


def prefill_lane_capacity(device):
    grid = device.compute_with_storage_grid_size()
    return grid.x * grid.y // _LANE_GROUPS


@lru_cache(maxsize=8)
def _lane_layout(device, lanes):
    """Assign 18 column groups to each lane and cover the lane with multicast rectangles."""
    bounds = device.compute_with_storage_grid_size()

    def phys(x, y):
        return device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))

    xs = [phys(x, 0).x for x in range(bounds.x)]
    ys = [phys(0, y).y for y in range(bounds.y)]
    runs, start = [], 0
    for x in range(1, bounds.x + 1):
        if x == bounds.x or xs[x] != xs[x - 1] + 1:
            runs.append(range(start, x))
            start = x
    order = [(x, y) for run in runs for y in range(bounds.y) for x in run]
    if lanes * _LANE_GROUPS > len(order):
        raise ValueError(f"{lanes} lanes need {lanes * _LANE_GROUPS} workers")
    layout = []
    for lane in range(lanes):
        cores = order[lane * _LANE_GROUPS : (lane + 1) * _LANE_GROUPS]
        rows = {}
        for x, y in cores:
            run = next(i for i, r in enumerate(runs) if x in r)
            rows.setdefault((run, y), []).append(x)
        rects = []
        for (run, y), cols in sorted(rows.items()):
            x0, x1 = min(cols), max(cols)
            if rects and rects[-1][:3] == [run, x0, x1] and rects[-1][4] == y - 1 and ys[y] == ys[y - 1] + 1:
                rects[-1][4] = y
            else:
                rects.append([run, x0, x1, y, y])
        flat = []
        for _, x0, x1, y0, y1 in rects:
            a, b = phys(x0, y0), phys(x1, y1)
            flat.extend((a.x, a.y, b.x, b.y, (x1 - x0 + 1) * (y1 - y0 + 1)))
        coords = tuple(v for x, y in cores for v in (phys(x, y).x, phys(x, y).y))
        for me, (x, y) in enumerate(cores):
            layout.append((x, y, me, lane, len(cores), len(rects), tuple(flat), coords))
    return tuple(layout)


@lru_cache(maxsize=16)
def _lane_tasks(device, lanes, gate, pairs):
    def banks_of(g):
        if gate:
            return (g // 2, g // 2, 0, 0) if g < 16 else (*pairs[2 * (g - 16)], *pairs[2 * (g - 16) + 1])
        return tuple((4 * g + i) % _BANKS for i in range(4))

    tasks = []
    for x, y, me, lane, count, nrect, rects, coords in _lane_layout(device, lanes):
        tasks.append((x, y, me, lane, lanes, *banks_of(me), (x + y) % 4, me, count, nrect, *rects, *coords))
    return tuple(tasks)


@lru_cache(maxsize=16)
def _lane_program(gate, tasks, fidelity, specs, wide_blocks, weight_buffers):
    k = _GU_K if gate else _DOWN_K
    block_k = _GU_K_BLOCK if gate else _DOWN_K_BLOCK
    cols = 4
    out_cols = cols // 2 if gate else cols
    rt = [(ttnn.CoreCoord(x, y), list(args)) for x, y, *args in tasks]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, _ in rt])
    cbs = [
        _cb(grid, 0, _BF, 2048, 2 * k),
        _cb(grid, 1, _BFP8, 1088, weight_buffers * k * cols),
        _cb(grid, 2, ttnn.uint32, _EXPERTS * 4),
        _cb(grid, 3, ttnn.uint32, _EXPERTS * 4),
        _cb(grid, 16, _BF, 2048, 2 * out_cols),
    ]
    semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=int(i == 2)) for i in range(3)]
    definitions = [("EXPERTS_GATE", str(int(gate))), ("EXPERTS_WIDE_L1_BLOCKS", str(wide_blocks))]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_PREFILL_KERNELS / source),
            core_ranges=grid,
            compile_time_args=[k, block_k, cols, *accessor],
            runtime_args=rt,
            defines=definitions,
            config=config,
        )
        for source, accessor, config in (
            ("lane_weights.cpp", specs[0], ttnn.ReaderConfigDescriptor()),
            ("lane_rows.cpp", sum(specs, ()), ttnn.WriterConfigDescriptor()),
            (
                "lane_matmul.cpp",
                (),
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=fidelity,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=gate,
                ),
            ),
        )
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernels


def _shared_pairs(handle):
    shared = sorted({(task[2] - 32, task[3], task[4]) for task in handle.gate_tasks[1]})
    return tuple((first, second) for _, first, second in shared)


def _prefill_project_lanes(gate, workspace, handle, compact=None):
    buffer, tail, wide_blocks = _compact(workspace, compact)
    x, output = (buffer, workspace.hidden) if gate else (workspace.hidden, buffer)
    weights = handle.gate_up if gate else handle.down
    tasks = _lane_tasks(x.device(), workspace.lanes, gate, _shared_pairs(handle))
    ends = workspace.ends
    tensors = (ends, x, output, *tail)
    with _LOCK:
        desc, kernels = _lane_program(
            gate,
            tasks,
            handle.fidelity,
            tuple(_accessor(t) for t in tensors),
            wide_blocks,
            workspace.weight_buffers,
        )
        kernels[0].common_runtime_args = [ends.buffer_address(), weights.buffer_address()]
        kernels[1].common_runtime_args = [t.buffer_address() for t in tensors]
        desc.kernels = kernels
        ttnn.generic_op([ends, x, weights, output, *tail], desc)


def moe_prefill(x, routing, handle, *, workspace=None):
    """Compute the BF16 [1,1,T,2304] MoE branch with the decode weight pack.

    x is interleaved BF16 TILE. Pass CPU (ids, gamma) of shape [valid_tokens,8]
    with FP32 gamma. Rows beyond valid_tokens return zero. For device-only
    replay, pass routing=None and a workspace with staged routing. Warm up all
    four programs on the same device before trace capture.
    """
    rowvm._validate(x, (1, 1, x.shape[-2], _WIDTH))
    if x.dtype != _BF:
        raise ValueError("expert prefill requires BF16 x")
    device = x.device()
    if any(t.device() != device for t in (handle.gate_up, handle.down)):
        raise ValueError("expert prefill tensors must be on the same device")
    if workspace is None:
        if routing is None:
            raise ValueError("routing=None requires a workspace with staged routing")
        workspace = prepare_prefill(device, x.shape[-2])
    if workspace.tokens != x.shape[-2] or workspace.output.device() != device:
        raise ValueError("workspace must match the input device and chunk size")
    if routing is not None:
        stage_prefill_routing(routing, workspace)
    if not workspace.lanes:
        _prefill_io(x, workspace, False)
        _prefill_project(True, workspace, handle)
        _prefill_project(False, workspace, handle)
        _prefill_io(x, workspace, True)
        return workspace.output
    _prefill_io(x, workspace, False)
    _prefill_project_lanes(True, workspace, handle)
    # Above 1,024 rows the gathered input stays in DRAM. Keep the down output
    # in L1 only between the down projection and the row-gathering combine.
    down_out = None
    if workspace.down_l1_blocks:
        down_out = _prefill_alloc(
            device,
            (1, 1, workspace.down_l1_blocks * _GU_K, workspace.rows * 32),
            _BF,
            _RM,
        )
    _prefill_project_lanes(False, workspace, handle, down_out)
    _prefill_io(x, workspace, True, down_out)
    return workspace.output
