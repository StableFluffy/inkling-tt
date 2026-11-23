"""Inkling-2-Mini projections tuned on one Blackhole p150a.

Weights are [K, N] on the host. Decode uses one logical row. Prefill shares
the decode weight buffer across all tile rows. Prepare weights before capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

import torch
import ttnn

from inkling_tt.grid import worker_grid

_PREFILL_KERNELS = Path(__file__).resolve().parent / "kernels" / "prefill"
_PREFILL_LOCK = Lock()


@dataclass(frozen=True)
class _Config:
    path: str
    fidelity: str
    block: int = 0
    workers: int = 0
    input_shard_tiles: int = 0
    output_shard_tiles: int = 0


# Keys are (K, N, weight dtype).
_CONFIGS = {
    (2304, 3072, "bf16"): _Config("pipe", "HiFi4", 2, 4, 4, 2),
    (2304, 3072, "bfp8"): _Config("1d", "HiFi2", 8, 0, 0, 1),
    (2304, 1024, "bf16"): _Config("1d", "HiFi4", 8, 0, 0, 1),
    (2304, 1024, "bfp8"): _Config("1d", "HiFi2", 24, 0, 0, 1),
    (3072, 2304, "bf16"): _Config("pipe", "HiFi4", 2, 4, 4, 2),
    (3072, 2304, "bfp8"): _Config("1d", "HiFi2", 8, 0, 0, 1),
    (2304, 384, "bf16"): _Config("1d", "HiFi4", 36, 0, 0, 1),
    (2304, 384, "bfp8"): _Config("1d", "HiFi2", 36, 0, 0, 1),
    (2304, 24, "bf16"): _Config("1d", "HiFi4", 24, 0, 0, 1),
    (2304, 24, "bfp8"): _Config("1d", "HiFi2", 36, 0, 0, 1),
    (2304, 32, "bf16"): _Config("1d", "HiFi4", 24, 0, 0, 1),
    (2304, 32, "bfp8"): _Config("1d", "HiFi2", 36, 0, 0, 1),
    (4608, 64, "bf16"): _Config("pipe", "HiFi4", 6, 8, 4, 1),
    (4608, 64, "bfp8"): _Config("pipe", "HiFi2", 2, 8, 4, 1),
    (2304, 128, "bf16"): _Config("1d", "HiFi4", 36, 0, 0, 1),
    (2304, 6560, "bf16"): _Config("pipe", "HiFi4", 2, 4, 4, 4),
    (2304, 6560, "bfp8"): _Config("pipe", "HiFi2", 2, 8, 4, 4),
    (2304, 2048, "bf16"): _Config("1d", "HiFi4", 4, 0, 0, 1),
    (2304, 2048, "bfp8"): _Config("1d", "HiFi2", 8, 0, 0, 1),
    (2304, 201024, "bf16"): _Config("pipe", "HiFi4", 2, 4, 4, 61),
    (2304, 201024, "bfp8"): _Config("pipe", "HiFi2", 4, 8, 4, 61),
}


@dataclass(frozen=True)
class DenseWeight:
    tensor: ttnn.Tensor
    k: int
    n: int
    output_dtype: ttnn.DataType
    program: object | None
    compute: object
    input_shards: ttnn.MemoryConfig | None
    output_shards: ttnn.MemoryConfig | None


@dataclass(frozen=True)
class FusedWeight:
    weight: DenseWeight
    widths: tuple[int, ...]


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def _width_shards(grid, rows: int, cols: int, buffer_type: ttnn.BufferType) -> ttnn.MemoryConfig:
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type,
        ttnn.ShardSpec(grid, (rows, cols), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _l1_shards(device, length: int, shard_tiles: int) -> ttnn.MemoryConfig:
    width = 32 * shard_tiles
    cores = ttnn.num_cores_to_corerangeset(
        _ceil_div(length, width),
        device.compute_with_storage_grid_size(),
        row_wise=True,
    )
    return _width_shards(cores, 32, width, ttnn.BufferType.L1)


@lru_cache(maxsize=16)
def _prefill_columns(device, n, k, block, decode_workers):
    bounds = device.compute_with_storage_grid_size()
    assignment = device.get_optimal_dram_bank_to_logical_worker_assignment(
        ttnn.NOC.NOC_0,
        ttnn.MeshCoordinate(0, 0),
    )
    preferred = [assignment[bank] for bank in sorted(assignment)]
    bank_width = _ceil_div(n, len(preferred))
    available = {(x, y) for y in range(bounds.y) for x in range(bounds.x)}
    tasks = []
    for bank, near in enumerate(preferred):
        for group in range(decode_workers):
            start = bank_width * group // decode_workers
            end = min(bank_width * (group + 1) // decode_workers, n - bank * bank_width)
            rotate = (bank * decode_workers + group) * (k // block) // (len(preferred) * decode_workers)
            width = 1 if n <= 72 else 2
            for lo in range(start, end, width):
                x, y = min(available, key=lambda p: ((p[0] - near.x) % bounds.x + 2 * abs(p[1] - near.y), p))
                available.remove((x, y))
                tasks.append((x, y, bank * bank_width + lo, min(end - lo, width), rotate))
    return tuple(tasks)


@lru_cache(maxsize=32)
def _prefill_resident_program(tasks, m, k, n, block, in_dtype, out_dtype, splits, interleaved, specs):
    in_bytes = 2048 if in_dtype == ttnn.bfloat16 else 4096
    out_bytes = 2048 if out_dtype == ttnn.bfloat16 else 4096
    all_cores = [ttnn.CoreCoord(x, y) for x, y, *_ in tasks]
    all_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in all_cores])

    def cb(grid, index, dtype, size, pages):
        return ttnn.CBDescriptor(
            total_size=size * pages,
            core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size),
            ],
        )

    input_pages = k if m >= 64 and k >= 72 else 2 * k
    kernels, cbs = [], [cb(all_grid, 0, in_dtype, in_bytes, input_pages)]
    widths = sorted({task[3] for task in tasks})
    for width in widths:
        rt = [(ttnn.CoreCoord(x, y), args) for x, y, *args in tasks if args[1] == width]
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, _ in rt])
        cbs.extend((cb(grid, 1, in_dtype, in_bytes, k * width), cb(grid, 16, out_dtype, out_bytes, 2 * width)))
        common = [k, n, m, block, in_bytes, out_bytes, splits, int(interleaved)]
        kernels.extend(
            [
                ttnn.KernelDescriptor(
                    kernel_source=str(_PREFILL_KERNELS / "dense_io.cpp"),
                    core_ranges=grid,
                    compile_time_args=[*common, *specs[2]],
                    runtime_args=rt,
                    defines=[("PREFILL_READ_WEIGHT", "1")],
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(_PREFILL_KERNELS / "dense_io.cpp"),
                    core_ranges=grid,
                    compile_time_args=[*common, *specs[0], *specs[1]],
                    runtime_args=rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(_PREFILL_KERNELS / "dense_compute.cpp"),
                    core_ranges=grid,
                    compile_time_args=common,
                    runtime_args=rt,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                    ),
                ),
            ]
        )
    semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_grid, initial_value=int(i == 2)) for i in range(3)]
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernels


@lru_cache(maxsize=16)
def _prefill_multicast(device, tasks):
    physical = {(x, y): device.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) for x, y, *_ in tasks}
    by_row = {}
    for logical, core in physical.items():
        by_row.setdefault(core.y, {})[core.x] = logical
    bindings = {}
    for y, row in by_row.items():
        start = previous = min(row)
        for x in (*sorted(row)[1:], max(row) + 2):
            if x != previous + 1:
                for col in range(start, previous + 1):
                    bindings[row[col]] = (start, previous, y, int(col == start))
                start = x
            previous = x
    return tuple((*task, *bindings[task[:2]]) for task in tasks)


def _linear_prefill_resident(x, weight, n, block, tasks, dtype, *, splits=1, interleaved=False):
    device = x.device()
    m, k = x.shape[-2] // 32, x.shape[-1] // (32 * splits)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, splits, m * 32, n * 32)),
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    with _PREFILL_LOCK:
        specs = tuple(tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in (x, out, weight))
        desc, kernels = _prefill_resident_program(tasks, m, k, n, block, x.dtype, dtype, splits, interleaved, specs)
        for reader, writer in zip(kernels[::3], kernels[1::3]):
            reader.common_runtime_args = [weight.buffer_address()]
            writer.common_runtime_args = [x.buffer_address(), out.buffer_address()]
        desc.kernels = kernels
        ttnn.generic_op([x, weight, out], desc)
    return out


def linear_prefill_resident(x, handle, dtype=ttnn.bfloat16):
    """Read each width-sharded weight tile once and retain it for all M tiles."""
    k, n = handle.k // 32, handle.n // 32
    program = handle.program
    block = program.in0_block_w
    columns = _prefill_columns(x.device(), n, k, block, program.num_workers_per_dram_bank)
    return _linear_prefill_resident(
        x,
        handle.tensor,
        n,
        block,
        _prefill_multicast(x.device(), columns),
        dtype,
    )


@lru_cache(maxsize=16)
def _message_down_workers(device, splits, blocks):
    grid = device.compute_with_storage_grid_size()
    return tuple(
        (i % grid.x, i // grid.x, col, 1, col * blocks // 8, 0, 0, 0, 1, split)
        for i in range(2 * splits)
        for split, col in (divmod(i, 2),)
    )


@lru_cache(maxsize=16)
def _message_reduce_program(tasks, tiles, splits, specs):
    cores = [ttnn.CoreCoord(i % tasks[0], i // tasks[0]) for i in range(tasks[1])]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
    rt = [(core, [i * tiles // len(cores), (i + 1) * tiles // len(cores)]) for i, core in enumerate(cores)]
    source = str(_PREFILL_KERNELS / "dense_reduce.cpp")
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=source,
            core_ranges=grid,
            runtime_args=rt,
            compile_time_args=[tiles, splits, *sum(specs, ())],
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=source,
            core_ranges=grid,
            runtime_args=rt,
            compile_time_args=[splits],
            defines=[("REDUCE_COMPUTE", "1")],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
            ),
        ),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=size * pages,
            core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size),
            ],
        )
        for index, dtype, size, pages in ((0, ttnn.float32, 4096, 2 * splits), (16, ttnn.bfloat16, 2048, 2))
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernels


def message_prefill_down(x, handle):
    """Share the decode K partitions across the chunk and reduce their FP32 partials."""
    splits = handle.program.num_workers_per_dram_bank
    block = handle.program.in0_block_w
    tasks = _message_down_workers(x.device(), splits, 144 // (block * splits))
    partial = _linear_prefill_resident(x, handle.tensor, 2, block, tasks, ttnn.float32, splits=splits)
    device = x.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, 1, x.shape[-2], 64)),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    grid = device.compute_with_storage_grid_size()
    tiles = x.shape[-2] // 16
    with _PREFILL_LOCK:
        specs = tuple(tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in (partial, out))
        desc, kernels = _message_reduce_program((grid.x, min(tiles, grid.x * grid.y)), tiles, splits, specs)
        kernels[0].common_runtime_args = [partial.buffer_address(), out.buffer_address()]
        desc.kernels = kernels
        ttnn.generic_op([partial, out], desc)
    return out


def message_prefill_up(x, weight):
    """Retain FP32 controller weights across all rows."""
    n = weight.shape[-1] // 32
    grid = x.device().compute_with_storage_grid_size()
    columns = tuple((i % grid.x, i // grid.x, i, 1, 0) for i in range(n))
    return _linear_prefill_resident(
        x,
        weight,
        n,
        1,
        _prefill_multicast(x.device(), columns),
        ttnn.float32,
        interleaved=True,
    )


def prepare_weight(device, w_torch: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat16) -> DenseWeight:
    """Upload one projection in the storage layout selected for its shape and dtype."""
    if w_torch.ndim != 2 or w_torch.device.type != "cpu" or not w_torch.is_floating_point():
        raise ValueError("Expected a floating-point CPU weight tensor with shape [K, N]")
    k, n = w_torch.shape
    dtype_name = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8"}.get(dtype)
    if (k, n) == (2304, 128) and dtype != ttnn.bfloat16:
        raise ValueError("The router requires BF16 weights")
    config = _CONFIGS.get((k, n, dtype_name))
    if config is None:
        raise ValueError(f"No tuned Inkling decode configuration for {k}x{n} with {dtype}")
    grid = worker_grid(device)
    dram_grid = device.dram_grid_size()
    if (dram_grid.x, dram_grid.y) != (8, 1):
        raise ValueError("Inkling dense projections require eight DRAM banks")

    weight_mem = ttnn.DRAM_MEMORY_CONFIG
    input_shards = output_shards = program = None
    if config.path in ("pipe", "ds"):
        banks = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        weight_mem = _width_shards(banks, k, 32 * _ceil_div(n, 256), ttnn.BufferType.DRAM)
        input_shards = _l1_shards(device, k, config.input_shard_tiles)
        output_shards = _l1_shards(device, n, config.output_shard_tiles)
        program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=config.block,
            per_core_M=1,
            per_core_N=config.output_shard_tiles,
            fused_activation=None,
            num_workers_per_dram_bank=config.workers,
            use_multicore_pipeline=config.path == "pipe",
        )
    elif config.path == "1d":
        width = config.output_shard_tiles
        subblock = max(d for d in range(1, min(width, 4) + 1) if width % d == 0)
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=config.block,
            out_subblock_h=1,
            out_subblock_w=subblock,
            out_block_h=1,
            out_block_w=width,
            per_core_M=1,
            per_core_N=width,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, config.fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    tensor = ttnn.from_torch(
        w_torch.detach().contiguous().view(1, 1, k, n),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weight_mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None,
    )
    default_output = ttnn.float32 if (k, n) in ((2304, 128), (4608, 64)) else ttnn.bfloat16
    return DenseWeight(tensor, k, n, default_output, program, compute, input_shards, output_shards)


def linear(
    x: ttnn.Tensor,
    handle: DenseWeight,
    *,
    memory_config: ttnn.MemoryConfig | None = None,
    output_dtype: ttnn.DataType | None = None,
) -> ttnn.Tensor:
    """Run a projection and return interleaved [1, 1, 1, N].

    Output stays in the input's L1 or DRAM buffer by default. The router
    accumulates BF16 products into FP32 Dest and returns FP32. Message-gate
    down rounds to BF16 first and then promotes to FP32 by default.
    """
    if tuple(x.shape) != (1, 1, 1, handle.k) or x.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"Expected a TILE activation of shape [1, 1, 1, {handle.k}]")
    if x.dtype not in (ttnn.bfloat16, ttnn.float32):
        raise ValueError("Expected a BF16 or FP32 activation")
    input_mem = x.memory_config()
    output_mem = input_mem if memory_config is None else memory_config
    interleaved = (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)
    if input_mem not in interleaved or output_mem not in interleaved:
        raise ValueError("The activation and the result require interleaved L1 or DRAM")
    if output_dtype is None:
        output_dtype = handle.output_dtype
    router = (handle.k, handle.n) == (2304, 128)
    if router and output_dtype != ttnn.float32:
        raise ValueError("The router requires FP32 output")
    if output_dtype == ttnn.bfloat8_b and (handle.k, handle.n) != (2304, 201024):
        raise ValueError("BFP8_B output is supported for the LM head")
    if output_dtype not in (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b):
        raise ValueError("Expected BF16, FP32 or LM-head BFP8_B output")
    if x.dtype == ttnn.float32:
        x = ttnn.typecast(x, ttnn.bfloat16, memory_config=input_mem)
    if handle.input_shards is not None:
        x = ttnn.to_memory_config(x, handle.input_shards)
    message_down = (handle.k, handle.n) == (4608, 64)
    mm_dtype = ttnn.bfloat16 if message_down and output_dtype == ttnn.float32 else output_dtype
    y = ttnn.linear(
        x,
        handle.tensor,
        program_config=handle.program,
        compute_kernel_config=handle.compute,
        memory_config=output_mem if handle.output_shards is None else handle.output_shards,
        dtype=mm_dtype,
    )
    if handle.output_shards is not None:
        y = ttnn.to_memory_config(y, output_mem)
    if mm_dtype != output_dtype:
        y = ttnn.typecast(y, output_dtype, memory_config=output_mem)
    return y


def prepare_decode_kv(device, k, v):
    """Pack each output column's K tiles together in bank-local DRAM."""
    packed = torch.cat((k, v), dim=-1).reshape(72, 32, 64, 32).permute(1, 2, 0, 3).reshape(1, 1, 32, -1)
    banks = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
    return ttnn.from_torch(
        packed.contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=_width_shards(banks, 32, 8 * 72 * 32, ttnn.BufferType.DRAM),
    )


def prepare_fused_weight(
    device,
    weights: tuple[torch.Tensor, ...],
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> FusedWeight:
    """Pack Q, relative bias, sink and gate, or K and V, in that order."""
    if not weights or any(w.ndim != 2 or w.shape[0] != 2304 for w in weights):
        raise ValueError("Expected [2304, N] weight tensors")
    widths = tuple(w.shape[-1] for w in weights)
    if widths not in ((3072, 384, 24, 3072), (1024, 1024)):
        raise ValueError("Expected Q/relative/sink/gate or K/V projection widths")
    padded = tuple(torch.nn.functional.pad(w, (0, (-n) % 32)) for w, n in zip(weights, widths))
    return FusedWeight(prepare_weight(device, torch.cat(padded, dim=-1), dtype), widths)


def linear_fused(
    x: ttnn.Tensor,
    handle: FusedWeight,
    *,
    memory_config: ttnn.MemoryConfig | None = None,
) -> tuple[ttnn.Tensor, ...]:
    """Apply the packed weights to one shared activation and return each projection."""
    y = linear(x, handle.weight, memory_config=memory_config)
    output_mem = y.memory_config()
    start = 0
    outputs = []
    for width in handle.widths:
        outputs.append(ttnn.slice(y, [0, 0, 0, start], [1, 1, 1, start + width], memory_config=output_mem))
        start += 32 * _ceil_div(width, 32)
    return tuple(outputs)
