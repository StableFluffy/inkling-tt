"""Fused batch-one Inkling-2-Mini attention and O-projection input on Blackhole."""

from functools import lru_cache
from pathlib import Path
from threading import Lock

import torch
import ttnn

from inkling_tt.grid import worker_grid

_KERNELS = Path(__file__).resolve().parent / "kernels" / "attn"
_SPLITS = 16
_KV_HEADS = 8
_WINDOW = 1024
_MAX_LENGTH = 32768
_LAUNCH_LOCK = Lock()


@lru_cache(maxsize=24)
def _program(accessors, length, global_cache, gate_tile, cols, rows, first_head, heads):
    grid = ttnn.num_cores_to_corerangeset(heads * _SPLITS, ttnn.CoreCoord(cols, rows), row_wise=True)
    split_tiles = (length // 32 + _SPLITS - 1) // _SPLITS
    tiles = min(split_tiles, 4)
    args = [length // 32, int(global_cache), gate_tile, cols, first_head]
    offset = len(args) + len(accessors)
    for accessor in accessors:
        args.append(offset)
        offset += len(accessor)
    for accessor in accessors:
        args.extend(accessor)

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=True,
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for index in (8, 10, 12, 13):
        unpack[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if split_tiles > 4:
        unpack[20] = unpack[21] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = unpack
    staged = len(accessors) > 9
    defines = [("ATTN_STAGED", "1")] if staged else []
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "io.cpp"),
            core_ranges=grid,
            compile_time_args=args,
            defines=defines,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "io.cpp"),
            core_ranges=grid,
            compile_time_args=args,
            defines=[*defines, ("ATTN_WRITER", "1")],
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "compute.cpp"),
            core_ranges=grid,
            compile_time_args=[length // 32, int(global_cache), cols],
            config=config,
        ),
    ]

    # CBs that act as NoC mailboxes use the same addresses on every worker.
    bf, fp = ttnn.bfloat16, ttnn.float32
    buffers = (
        (0, bf, 4, 2048),
        (1, bf, 4 * tiles, 2048),
        (2, bf, 4 * tiles, 2048),
        (3, bf, 1, 2048),
        (4, bf, tiles, 2048),
        (5, fp, 1, 64),
        (6, fp, 1, _SPLITS * 32),
        (7, fp, 1, 4096),
        (8, fp, 2, 4096),
        (9, fp, 1, 4096),
        (10, fp, 1, 4096),
        (11, bf, 1, 256),
        (12, fp, split_tiles, 4096),
        (13, fp, _SPLITS, 4096),
        (14, bf, 1, 2048),
        (15, bf, tiles, 2048),
        (16, fp, 4, 4096),
        (17, bf, 1, 2048),
        (18, fp, 1, 32),
        (19, fp, 1, 32),
    )
    if split_tiles > 4:
        buffers += ((20, fp, 1, 4096), (21, fp, 4, 4096))
    if staged:
        buffers += ((22, bf, 1, 2048), (23, bf, 1, 2048))
    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * size,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(index, fmt, size)],
        )
        for index, fmt, pages, size in buffers
    ]
    semaphores = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, grid, 0) for i in range(3)]
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernels


@lru_cache(maxsize=4)
def _placement(device, kv_heads):
    grid = worker_grid(device)
    # A smaller grid runs independent head groups in separate launches. Each
    # head retains the same split boundaries and FP32 reduction order.
    heads_per_launch = kv_heads if kv_heads * _SPLITS <= grid.x * grid.y else kv_heads // 2
    cores = [
        device.worker_core_from_logical_core(ttnn.CoreCoord(i % grid.x, i // grid.x))
        for i in range(heads_per_launch * _SPLITS)
    ]
    peers = tuple(c.x | (c.y << 16) for c in cores)
    return grid.x, grid.y, heads_per_launch, peers


def _table_height(global_cache):
    return 32 * (_WINDOW // 32 + int(global_cache)) * 32


def prepare_table(table, *, global_cache, device=None):
    """Pack all 32 position phases before capture so each lookup is one tile read."""
    if device is None:
        if table.dtype != ttnn.bfloat16:
            raise ValueError("fused attention decode requires a BF16 relative table")
        device = table.device()
        table = ttnn.to_torch(table)
    if tuple(table.shape) != (_WINDOW + 1, 32):
        raise ValueError("fused attention decode requires a relative table of shape [1025, 32]")
    rows = table.bfloat16()
    if global_cache and torch.count_nonzero(rows[-1]):
        raise ValueError("global fused attention requires a zero relative-bias sentinel")
    blocks = _WINDOW // 32 + int(global_cache)
    phases = torch.arange(32).reshape(-1, 1, 1)
    block_offsets = torch.arange(blocks).reshape(1, -1, 1) * 32
    distance = phases + block_offsets - torch.arange(32)
    if global_cache:
        index = torch.where((distance >= 0) & (distance < _WINDOW), distance, _WINDOW)
    else:
        index = distance % _WINDOW
    packed = rows[index].reshape(1, 1, _table_height(global_cache), 32)
    return ttnn.from_torch(
        packed,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def decode(q, k, v, relative, sink, table, position, gate, *, global_cache, gate_col, out, staged=None):
    """Apply position-derived attention and write the gated row into preallocated O shards."""
    length = k.shape[-2]
    if length < 32 or length > _MAX_LENGTH or length % 32 or (not global_cache and length != _WINDOW):
        raise ValueError("fused attention needs a 1024-slot ring or a tile-aligned global cache up to 32768 slots")
    kv_heads = k.shape[1]
    if kv_heads not in (4, _KV_HEADS):
        raise ValueError("fused attention decode supports four or eight KV heads per rank")
    query_heads = kv_heads * 3
    width = query_heads * 128
    if gate_col < 0 or gate_col % 32 or gate.shape[-1] < gate_col + width:
        raise ValueError("fused attention decode requires a tile-aligned output gate")
    device = q.device()
    inputs = (q, k, v, relative, sink, table, position, gate)
    operands = (*inputs, out)
    shapes = (
        (1, 1, query_heads, 128),
        (1, kv_heads, length, 128),
        (1, kv_heads, length, 128),
        (1, 1, query_heads, 32),
        (32, 32),
        (1, 1, _table_height(global_cache), 32),
        (1,),
        (1, 1, 1, gate.shape[-1]),
        (1, 1, 1, width),
    )
    for i, (operand, shape) in enumerate(zip(operands, shapes)):
        dtype = ttnn.int32 if i == 6 else ttnn.bfloat16
        layout = ttnn.ROW_MAJOR_LAYOUT if i == 6 else ttnn.TILE_LAYOUT
        if (
            not operand.is_allocated()
            or tuple(operand.shape) != shape
            or operand.device() != device
            or operand.dtype != dtype
            or operand.layout != layout
            or (i < 7 and operand.is_sharded())
        ):
            raise ValueError(f"invalid fused attention decode operand {i}")
    if (
        gate.memory_config().buffer_type != ttnn.BufferType.L1
        or out.memory_config().buffer_type != ttnn.BufferType.L1
        or out.memory_config().memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED
    ):
        raise ValueError("fused attention decode requires L1 gate logits and O-projection shards")
    if out.buffer_unique_id() in {t.buffer_unique_id() for t in inputs}:
        raise ValueError("fused attention decode requires independent output storage")

    if staged is not None:
        staged_k, staged_v, start = staged
        for state in (staged_k, staged_v):
            if (
                tuple(state.shape) != (1, kv_heads, 32, 128)
                or state.dtype != ttnn.bfloat16
                or state.layout != ttnn.TILE_LAYOUT
                or state.is_sharded()
                or state.device() != device
            ):
                raise ValueError("staged attention K/V must be interleaved BF16 tiles for the current block")
        if (
            tuple(start.shape) != (1,)
            or start.dtype != ttnn.int32
            or start.layout != ttnn.ROW_MAJOR_LAYOUT
            or start.device() != device
        ):
            raise ValueError("staged attention requires the block start position on device")
        operands = (*operands, *staged)
    accessors = tuple(tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in operands)
    cols, rows, heads, peers = _placement(device, kv_heads)
    addresses = [t.buffer_address() for t in operands]
    with _LAUNCH_LOCK:
        for first_head in range(0, kv_heads, heads):
            desc, kernels = _program(accessors, length, global_cache, gate_col // 32, cols, rows, first_head, heads)
            kernels[0].common_runtime_args = [*addresses, *peers]
            kernels[1].common_runtime_args = [*addresses, *peers]
            desc.kernels = kernels
            ttnn.generic_op(list(operands), desc)
    return out
