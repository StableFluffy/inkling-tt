"""Batch-one Inkling decode epilogues on Blackhole."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from struct import pack, unpack
from threading import Lock

import ttnn

FP = ttnn.float32
BF = ttnn.bfloat16
TILE = ttnn.TILE_LAYOUT
_KERNELS = Path(__file__).resolve().parent / "kernels" / "rowvm"
_RECT = os.environ.get("INK_ROWVM_RECT", "1") != "0"
_MIX_CHUNK = int(os.environ.get("INK_MIX_CHUNK", "128"))
_ZONES = [("ROWVM_ZONES", "1")] if os.environ.get("INK_ROWVM_ZONES") == "1" else []

(
    _RMS,
    _L2,
    _ADD,
    _GATE,
    _CONV,
    _MOE_HIDDEN,
    _MOE_COMBINE,
    _GATE_INPUT,
    _MESSAGE,
    _MIX,
    _ROUTER,
    _RMS_STRIPED,
    _GATE_INPUT_STRIPED,
    _MESSAGE_STRIPED,
    _SINK_TILE,
    _CONV_ADD,
    _CONV_MIX,
    _MESSAGE_UP,
    _CONV_K,
    _CONV_V,
) = range(20)
_MIX_OPS = (_MIX, _CONV_MIX)
_KV_OPS = (_CONV_K, _CONV_V)
_COLLECTIVES = (*_MIX_OPS, _RMS_STRIPED, _GATE_INPUT_STRIPED, _MESSAGE_STRIPED, _MESSAGE_UP)


@dataclass(frozen=True)
class _Region:
    tensor: ttnn.Tensor
    cols: int
    rows: int = 1
    row: int = 0
    col: int = 0
    row_step: int = 0
    col_step: int = 0
    concat: bool = False
    group_col_step: int = 0
    extent: int = 0
    shared_prefix: bool = False
    whole_tiles: bool = False
    cache_length: int = 0

    def spec(self, cb):
        height = self.tensor.shape[-2]
        batches = self.tensor.shape[1] if len(self.tensor.shape) == 4 else 1
        if batches != 1 and height != 1 and not self.cache_length:
            raise ValueError("rowvm requires [1,1,H,W] or [1,H,1,W]")
        if any(v % 16 for v in (self.cols, self.col, self.col_step, self.group_col_step, self.extent)):
            raise ValueError("rowvm regions must be aligned to 16 columns")
        fmt = 4 if self.tensor.dtype == FP else 2
        padded_cols = self.tensor.padded_shape[-1]
        row_major = self.tensor.layout == ttnn.ROW_MAJOR_LAYOUT
        if row_major and (
            self.whole_tiles
            or self.cache_length
            or padded_cols % 32
            or any(v * fmt % 64 for v in (self.col, self.col_step, self.group_col_step, self.cols))
        ):
            # DRAM transfers keep source and destination aligned modulo 64 bytes.
            raise ValueError("row-major rowvm regions support 64-byte aligned row windows only")
        return (
            cb,
            fmt,
            padded_cols // 32,
            int(batches != 1),
            self.rows,
            self.cols,
            self.row_step,
            self.col_step // 16,
            int(self.concat),
            self.group_col_step // 16,
            self.extent,
            int(self.shared_prefix),
            int(self.whole_tiles),
            self.cache_length,
            int(row_major),
            tuple(ttnn.TensorAccessorArgs(self.tensor).get_compile_time_args()),
        )


def _validate(tensor, shape=None, *, rank=4, allow_sharded=False):
    if tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
        # Row-major L1 rows: one page per logical row, read with one transfer per stripe.
        if (
            len(tensor.shape) != rank
            or (rank == 4 and tensor.shape[0] != 1)
            or tensor.dtype not in (FP, BF)
            or tensor.is_sharded()
            or not tensor.is_allocated()
            or tensor.device() is None
        ):
            raise ValueError("rowvm row-major tensors must be interleaved FP32 or BF16 rows")
        if shape is not None and tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"expected shape {tuple(shape)}, got {tuple(tensor.shape)}")
        return
    if (
        len(tensor.shape) != rank
        or (rank == 4 and tensor.shape[0] != 1)
        or tensor.layout != TILE
        or tensor.dtype not in (FP, BF)
        or (tensor.is_sharded() and not allow_sharded)
        or not tensor.is_allocated()
        or tensor.device() is None
    ):
        raise ValueError("rowvm requires a device FP32 or BF16 TILE tensor with the requested rank and placement")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"expected shape {tuple(shape)}, got {tuple(tensor.shape)}")
    tile = tensor.spec.tile
    if tile.tile_shape != [32, 32] or tile.transpose_within_face or tile.transpose_of_faces:
        raise ValueError("rowvm requires standard 32x32 tiles")


def _output(x, shape, dtype, out, memory_config, *, allow_sharded=False):
    if dtype not in (FP, BF):
        raise ValueError("rowvm output dtype must be float32 or bfloat16")
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), dtype, TILE, x.device(), memory_config or x.memory_config()
        )
    _validate(out, shape, rank=len(shape), allow_sharded=allow_sharded)
    if out.dtype != dtype or (memory_config is not None and out.memory_config() != memory_config):
        raise ValueError("preallocated rowvm output dtype or memory configuration does not match")
    return out


def _cb(grid, index, fmt, pages=1, page_size=None):
    size = page_size or 1024 * fmt
    return ttnn.CBDescriptor(
        total_size=pages * size,
        core_ranges=grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=FP if fmt == 4 else BF, page_size=size)
        ],
    )


def _io_args(specs, task_cols):
    args = [len(specs)]
    for spec in specs:
        args.extend(spec[:-1])
    offset = len(args) + len(specs) + 1
    for spec in specs:
        args.append(offset)
        offset += len(spec[-1])
    args.append(task_cols)
    for spec in specs:
        args.extend(spec[-1])
    return args


def _rects(core_coords):
    spans = []
    for x, y in sorted(core_coords, key=lambda c: (c[1], c[0])):
        if spans and spans[-1][2] == y and spans[-1][1] == x - 1:
            spans[-1] = (spans[-1][0], x, y)
        else:
            spans.append((x, x, y))
    rects = []
    for x0, x1, y in spans:
        if rects and rects[-1][:2] == (x0, x1) and rects[-1][3] == y - 1:
            rects[-1] = (x0, x1, rects[-1][2], y)
        else:
            rects.append((x0, x1, y, y))
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, c), ttnn.CoreCoord(b, d)) for a, b, c, d in rects])


def _linear_base(core_coords, grid_x):
    base = core_coords[0][1] * grid_x + core_coords[0][0]
    if all(y * grid_x + x == base + i for i, (x, y) in enumerate(core_coords)):
        return base
    return None


@lru_cache(maxsize=256)
def _program(compute_args, readers, writers, core_coords, task_cols, index_args, position_args, grid_width):
    cores = tuple(ttnn.CoreCoord(x, y) for x, y in core_coords)
    base = _linear_base(core_coords, grid_width) if _RECT else None
    grid = _rects(core_coords) if base is not None else ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    cbs = [
        _cb(grid, spec[0], spec[1], spec[4] * (spec[5] // 32 if spec[12] else (spec[5] + 1023) // 1024))
        for spec in readers + writers
    ]
    if compute_args[0] == _SINK_TILE:
        src, dst = readers[0][-1], writers[0][-1]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "sink_tile.cpp"),
            core_ranges=grid,
            compile_time_args=[1 + len(src), *src, *dst],
            config=ttnn.WriterConfigDescriptor(),
        )
        desc = ttnn.ProgramDescriptor(kernels=[kernel], cbs=cbs)
        desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
        return desc, [kernel]
    bf_cols = max((s[5] for s in readers if s[1] == 2), default=0)
    if bf_cols:
        cbs.append(_cb(grid, 31, 4, page_size=((bf_cols + 31) // 32) * 64))
    semaphores = []
    task_defines = _ZONES + (
        []
        if base is None
        else [
            ("ROWVM_TASK_BASE", str(base)),
            ("ROWVM_GRID_X", str(grid_width)),
        ]
    )
    defines = list(task_defines)
    if compute_args[0] in _COLLECTIVES:
        op, _, flags, _, _, stripes = compute_args
        is_mix = op in _MIX_OPS
        pages = (stripes + 15) // 16 if is_mix else 1
        for index, num_pages in ((8, pages), (9, 1), (18, 1), (19, 1)):
            cbs.append(_cb(grid, index, 4, num_pages))
        second_pass = bool(is_mix and flags & 1)
        append_bank = op == _CONV_MIX and bool(flags & 32)
        if second_pass or op in (_MESSAGE_STRIPED, _MESSAGE_UP):
            cbs.append(_cb(grid, 20, 4))
        semaphores = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, grid, 0) for i in (0, 1)]
        defines = [
            *task_defines,
            ("ROWVM_COLLECTIVE", "1"),
            ("ROWVM_STRIPES", str(stripes)),
            ("ROWVM_AGG_PAGES", str(pages)),
            ("ROWVM_OUTPUTS", str(len(cores) // stripes if is_mix else 0)),
            ("ROWVM_SECOND_PASS", str(int(second_pass))),
            ("ROWVM_RMS_ONLY", str(int(not is_mix))),
            ("ROWVM_APPEND_BANK", str(int(append_bank))),
        ]
        if is_mix:
            defines.append(("ROWVM_SPLIT_OUTPUT", "1"))
        if op == _CONV_MIX:
            cbs.append(_cb(grid, 10, 4))
            defines.append(("ROWVM_CONV_MIX", "1"))
        if append_bank:
            cbs.append(_cb(grid, 11, 4))
        if op == _MESSAGE_UP:
            defines.append(("ROWVM_MESSAGE_UP", "1"))
            cbs.extend((_cb(grid, 2, 4), _cb(grid, 3, 4), _cb(grid, 15, 4, 4)))
        if op == _GATE_INPUT_STRIPED:
            defines.append(("ROWVM_GROUP_INPUT", "1"))
    write_args = _io_args(writers, task_cols)
    write_defines = [*defines, ("ROWVM_WRITE", "1")]
    if compute_args[0] in (_RMS, _L2) and writers[0][5] > compute_args[1]:
        write_defines.append(("ROWVM_PAD_HEAD", "1"))
    if index_args:
        cbs.extend((_cb(grid, 8, 4), _cb(grid, 18, 4, 2)))
        write_defines.extend([("ROWVM_ROUTER", "1"), ("ROWVM_INDEX_ARGS", str(len(write_args)))])
        write_args.extend(index_args)
    if position_args:
        cbs.extend((_cb(grid, 10, 2), _cb(grid, 30, 4, page_size=64)))
        write_defines.extend(
            [
                ("ROWVM_CACHE_POSITION_ARGS", str(len(write_args))),
                ("ROWVM_CACHE_WRAP", str(compute_args[4])),
            ]
        )
        write_args.extend(position_args)
    rt_args = [(c, [task]) for task, c in enumerate(cores)] if base is None else []
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=True,
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for cb, fmt, *_ in readers:
        if fmt == 4 and not (compute_args[0] == _MESSAGE_UP and cb in (12, 13, 14)):
            unpack[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if compute_args[0] in _COLLECTIVES:
        for cb in (8, 9, 20):
            unpack[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if compute_args[0] == _CONV_MIX:
        unpack[10] = unpack[11] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if compute_args[0] == _MESSAGE_UP:
        unpack[2] = unpack[3] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if index_args:
        unpack[8] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode = unpack
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "io.cpp"),
            core_ranges=grid,
            compile_time_args=_io_args(readers, task_cols),
            runtime_args=rt_args,
            defines=defines,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "io.cpp"),
            core_ranges=grid,
            compile_time_args=write_args,
            runtime_args=rt_args,
            defines=write_defines,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "compute.cpp"),
            core_ranges=grid,
            compile_time_args=[*compute_args, *([0] * (6 - len(compute_args)))],
            runtime_args=rt_args,
            defines=task_defines,
            config=compute,
        ),
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernels


_launch_lock = Lock()


def _bind(kernel, regions, extra_args=()):
    kernel.common_runtime_args = (
        [r.tensor.buffer_address() for r in regions]
        + [value for r in regions for value in (r.row, r.col // 16)]
        + list(extra_args)
    )


def _run(
    compute_args, readers, writers, tasks=1, task_cols=0, query_outputs=(), indices=None, position=None, group=None
):
    inputs = [r.tensor for r in readers]
    if position is not None:
        inputs.append(position)
    outputs = [r.tensor for r in writers]
    if query_outputs:
        outputs = [*query_outputs, *outputs[1:]]
    if indices is not None:
        outputs.append(indices)
    device = inputs[0].device()
    if any(t.device() != device for t in inputs + outputs):
        raise ValueError("rowvm tensors must be on the same device")
    output_ids = [t.buffer_unique_id() for t in outputs]
    input_ids = {t.buffer_unique_id() for t in inputs}
    writable_ids = set(output_ids)
    if compute_args[0] in (_CONV, _CONV_ADD, _CONV_MIX, *_KV_OPS):
        writable_ids.remove(writers[1].tensor.buffer_unique_id())
    if compute_args[0] == _CONV_MIX and compute_args[2] & 32:
        writable_ids.remove(writers[-1].tensor.buffer_unique_id())
    if len(set(output_ids)) != len(output_ids) or input_ids.intersection(writable_ids):
        raise ValueError("rowvm outputs must have independent storage")
    grid = device.compute_with_storage_grid_size()
    offset = group.tasks if group is not None else 0
    if offset + tasks > grid.x * grid.y:
        raise ValueError("rowvm requires more logical cores than the device provides")
    coords = tuple((i % grid.x, i // grid.x) for i in range(offset, offset + tasks))
    in_cbs = (0, 1, 4, 12, 13, 14) if compute_args[0] == _MESSAGE_UP else range(len(readers))
    in_specs = tuple(r.spec(cb) for cb, r in zip(in_cbs, readers))
    out_cbs = (16, 17, 21, 22) if compute_args[0] == _CONV_MIX else range(16, 16 + len(writers))
    out_specs = tuple(r.spec(cb) for cb, r in zip(out_cbs, writers))
    common_args = []
    if compute_args[0] in _COLLECTIVES:
        if query_outputs:
            common_args.extend(t.buffer_address() for t in query_outputs)
        for x, y in coords:
            virtual = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            common_args.append(virtual.x | (virtual.y << 16))
    index_args = ()
    if indices is not None:
        index_args = tuple(ttnn.TensorAccessorArgs(indices).get_compile_time_args())
        common_args.append(indices.buffer_address())
    position_args = ()
    if position is not None:
        position_args = tuple(ttnn.TensorAccessorArgs(position).get_compile_time_args())
        common_args.append(position.buffer_address())
    args = (compute_args, in_specs, out_specs, coords, task_cols, index_args, position_args, grid.x)
    if group is not None:
        group.tasks += tasks
        group.entries.append((args, readers, writers, common_args, inputs, outputs))
        return outputs[0]
    with _launch_lock:
        desc, kernels = _program(*args)
        _bind_program(compute_args[0], kernels, readers, writers, common_args)
        desc.kernels = kernels
        ttnn.generic_op(inputs + outputs[::-1], desc)
    return outputs[0]


def _bind_program(op, kernels, readers, writers, common_args):
    if op == _SINK_TILE:
        _bind(kernels[0], readers + writers)
    else:
        _bind(kernels[0], readers, common_args)
        _bind(kernels[1], writers, common_args)


@lru_cache(maxsize=32)
def _group_program(keys):
    programs = [_program(*key) for key in keys]
    desc = ttnn.ProgramDescriptor(
        kernels=[k for _, kernels in programs for k in kernels],
        cbs=[cb for p, _ in programs for cb in p.cbs],
        semaphores=[s for p, _ in programs for s in p.semaphores],
    )
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, programs


class _Group:
    """One generic launch for independent kernels on disjoint workers."""

    def __init__(self):
        self.tasks = 0
        self.entries = []

    def launch(self):
        inputs, outputs, kernels = [], [], []
        with _launch_lock:
            desc, programs = _group_program(tuple(entry[0] for entry in self.entries))
            for (args, readers, writers, extra, ins, outs), (_, items) in zip(self.entries, programs):
                _bind_program(args[0][0], items, readers, writers, extra)
                kernels.extend(items)
                inputs.extend(ins)
                outputs.extend(outs)
            desc.kernels = kernels
            ttnn.generic_op(inputs + outputs[::-1], desc)


def rms_norm(x, weight=None, *, dtype=FP, out=None, memory_config=None):
    """Normalize a logical row, optionally multiply by a weight row and round to BF16."""
    _validate(x)
    width = x.shape[-1]
    if tuple(x.shape)[:3] != (1, 1, 1) or width not in range(32, 4609, 16):
        raise ValueError("rowvm RMSNorm requires a [1,1,1,N] row with 32 <= N <= 4608, aligned to 16")
    if weight is not None:
        _validate(weight, x.shape)
    flags = int(weight is not None) | (2 if dtype == BF else 0)
    y = _output(x, x.shape, dtype, out, memory_config)
    if width >= 1024:
        chunk, stripes = _stripes(width)
        operands = (x,) if weight is None else (x, weight)
        return _run(
            (_RMS_STRIPED, chunk, flags, 0, width, stripes),
            [_stripe(t, chunk, width) for t in operands],
            [_stripe(y, chunk, width)],
            stripes,
        )
    readers = [_Region(x, width)]
    if weight is not None:
        readers.append(_Region(weight, width))
    return _run((_RMS, width, flags), readers, [_Region(y, width)])


def head_norm(
    x, *, threshold=None, scale=None, tau=None, round_before_tau=True, dtype=BF, out=None, memory_config=None
):
    """Per-head RMSNorm or scaled L2 clip, with optional post-clip tau."""
    _validate(x)
    _, batch, heads, dim = x.shape
    if batch != 1 or heads not in (8, 24) or dim not in (16, 128):
        raise ValueError("rowvm head_norm requires [1,1,H,d] with H in {8,24} and d in {16,128}")
    if threshold is None and (scale is not None or tau is not None):
        raise ValueError("scale and tau are supported on the L2 clip path")
    readers = [_Region(x, dim, row_step=1)]
    flags = 2 if dtype == BF else 0
    if scale is not None:
        _validate(scale, x.shape)
        readers.append(_Region(scale, dim, row_step=1))
        flags |= 1
    if tau is not None:
        _validate(tau, (1, 1, 1, 1))
        readers.append(_Region(tau, 16))
        flags |= 4 | (8 if round_before_tau else 0)
    y = _output(x, x.shape, dtype, out, memory_config)
    args = (_RMS, dim, flags) if threshold is None else (_L2, dim, flags, _bits(threshold))
    return _run(args, readers, [_Region(y, dim, row_step=1)], heads)


def _row_window(x, width, col_offset, *, allow_sharded=False):
    _validate(x, allow_sharded=allow_sharded)
    if tuple(x.shape)[:3] != (1, 1, 1) or col_offset < 0 or col_offset % 16 or col_offset + width > x.shape[-1]:
        raise ValueError("rowvm requires a row window within [1,1,1,N] with a 16-aligned column offset")


def head_norm_from_row(
    x,
    heads,
    dim,
    *,
    threshold=None,
    tau=None,
    out_shape=None,
    col_offset=0,
    round_before_tau=True,
    dtype=BF,
    out=None,
    memory_config=None,
):
    """Write [1,1,H,d] from a 16-aligned window in a flat tile row.

    Use RMSNorm by default or L2 clip when threshold is set. Tau applies after
    clipping. A width-32 output for width-16 heads also writes their zero extension.
    """
    if heads not in (8, 24) or dim not in (16, 128):
        raise ValueError("rowvm head_norm_from_row requires H in {8,24} and d in {16,128}")
    _row_window(x, heads * dim, col_offset)
    shape = (1, 1, heads, dim) if out_shape is None else tuple(out_shape)
    if shape != (1, 1, heads, dim) and not (dim == 16 and shape == (1, 1, heads, 32)):
        raise ValueError("rowvm head_norm_from_row output must match the heads or zero-extend width 16 to 32")
    y = _output(x, shape, dtype, out, memory_config)
    return _norm_row_heads(x, y, dim, threshold, tau=tau, round_before_tau=round_before_tau, col_offset=col_offset)


def _norm_row_heads(x, y, dim, threshold, *, tau=None, round_before_tau=True, col_offset=0, group=None):
    readers = [_Region(x, dim, col=col_offset, col_step=dim)]
    flags = 2 if y.dtype == BF else 0
    if tau is not None:
        if threshold is None:
            raise ValueError("tau is supported on the L2 clip path")
        _validate(tau, (1, 1, 1, 1))
        readers.append(_Region(tau, 16))
        flags |= 4 | (8 if round_before_tau else 0)
    args = (_RMS, dim, flags) if threshold is None else (_L2, dim, flags, _bits(threshold))
    return _run(args, readers, [_Region(y, y.shape[-1], row_step=1)], y.shape[-2], group=group)


def kv_norm_to_shard(x, *, threshold=None, memory_config, out=None):
    """Write BF16 [1,1,8,128] directly to the L1 shard for a cache update.

    Pass the K clip threshold for L2 clipping. Omit it for V RMSNorm.
    """
    _validate(x, (1, 1, 1, 1024))
    shard = memory_config.shard_spec
    if (
        memory_config.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        or memory_config.buffer_type != ttnn.BufferType.L1
        or shard is None
        or tuple(shard.shape) != (32, 128)
        or shard.grid.num_cores() != 1
    ):
        raise ValueError("rowvm KV output requires one height-sharded 32x128 L1 shard")
    y = _output(x, (1, 1, 8, 128), BF, out, memory_config, allow_sharded=True)
    return _norm_row_heads(x, y, 128, threshold)


def sink_tile(x, *, col_offset=0, out=None):
    """Pack 24 already-scaled BF16 sink logits into column zero of a DRAM tile."""
    _row_window(x, 24, col_offset)
    if x.dtype != BF:
        raise ValueError("rowvm sink_tile requires BF16 logits")
    y = _output(x, (32, 32), BF, out, ttnn.DRAM_MEMORY_CONFIG)
    return _run((_SINK_TILE,), [_Region(x, 32, col=col_offset)], [_Region(y, 32)])


def attention_heads(x, q, r, sink, *, q_threshold, r_threshold, tau=None, _group=None):
    """Produce Q, zero-extended R, and sink tiles from the packed projection."""
    _validate(q, (1, 1, 24, 128))
    _validate(r, (1, 1, 24, 32))
    _validate(sink, (32, 32), rank=2)
    r_col = 24 * 128
    sink_col = r_col + 24 * 16
    _row_window(x, sink_col + 32, 0, allow_sharded=True)
    if any(t.dtype != BF for t in (x, q, r, sink)):
        raise ValueError("packed attention head preparation requires BF16 tensors")
    group = _Group() if _group is None else _group
    _norm_row_heads(x, q, 128, q_threshold, tau=tau, group=group)
    _norm_row_heads(x, r, 16, r_threshold, tau=tau, col_offset=r_col, group=group)
    _run((_SINK_TILE,), [_Region(x, 32, col=sink_col)], [_Region(sink, 32)], group=group)
    if _group is None:
        group.launch()


def update_kv(
    k,
    v,
    k_cache,
    v_cache,
    k_history,
    v_history,
    k_taps,
    v_taps,
    position,
    *,
    threshold,
    k_dyn=None,
    v_dyn=None,
    wrap,
    _group=None,
):
    """Convolve and normalize each head, then update the corresponding cache row."""
    if (
        tuple(position.shape) != (1,)
        or position.layout != ttnn.ROW_MAJOR_LAYOUT
        or position.dtype != ttnn.int32
        or position.is_sharded()
        or not position.is_allocated()
    ):
        raise ValueError("rowvm cache update requires a row-major int32 position")
    group = _Group() if _group is None else _group
    for op, x, cache, history, taps, dyn in (
        (_CONV_K, k, k_cache, k_history, k_taps, k_dyn),
        (_CONV_V, v, v_cache, v_history, v_taps, v_dyn),
    ):
        _validate(x, (1, 1, 1, 1024))
        _validate(history, (1, 1, 4, 1024))
        _validate(taps, history.shape)
        length = cache.shape[-2]
        _validate(cache, (1, 8, length, 128))
        if length < 32 or length % 32 or (wrap and length & (length - 1)) or cache.dtype != BF:
            raise ValueError("invalid BF16 cache shape or ring length")
        if history.dtype != FP:
            raise ValueError("rowvm cache convolution history must use float32")
        readers = [_stripe(t, 128, 1024, rows=n) for t, n in ((x, 1), (history, 3), (taps, 4))]
        if dyn is not None:
            _validate(dyn, (1, 1, 1, 32))
            readers.append(_Region(dyn, 32))
        _run(
            (op, 128, 2 | (4 if dyn is not None else 0), _bits(threshold), int(wrap)),
            readers,
            [_Region(cache, 128, cache_length=length), _stripe(history, 128, 1024, rows=4)],
            8,
            position=position,
            group=group,
        )
    if _group is None:
        group.launch()


def prepare_attention(
    qrsg, q, r, sink, k, v, caches, histories, taps, position, *, q_threshold, r_threshold, tau, k_dyn, v_dyn, wrap
):
    """Dispatch head preparation and the independent K/V cache writes together."""
    group = _Group()
    attention_heads(qrsg, q, r, sink, q_threshold=q_threshold, r_threshold=r_threshold, tau=tau, _group=group)
    update_kv(
        k,
        v,
        *caches,
        *histories,
        *taps,
        position,
        threshold=q_threshold,
        k_dyn=k_dyn,
        v_dyn=v_dyn,
        wrap=wrap,
        _group=group,
    )
    group.launch()


def gate_from_heads(x, logits, *, col_offset=0, dtype=BF, out=None, memory_config=None):
    """Multiply [1,1,H,d] by sigmoid of the logit window and write [1,1,1,H*d]."""
    _validate(x)
    _, batch, heads, dim = x.shape
    if batch != 1 or heads not in (8, 24) or dim not in (16, 128):
        raise ValueError("rowvm gate_from_heads requires [1,1,H,d] with H in {8,24} and d in {16,128}")
    width = heads * dim
    _row_window(logits, width, col_offset, allow_sharded=True)
    y = _output(x, (1, 1, 1, width), dtype, out, memory_config, allow_sharded=True)
    args = (_GATE, dim, 2 if dtype == BF else 0)
    return _run(
        args,
        [_Region(x, dim, row_step=1), _Region(logits, dim, col=col_offset, col_step=dim)],
        [_Region(y, dim, col_step=dim)],
        heads,
    )


@lru_cache(maxsize=64)
def _relative_program(
    accessors, index_page, tile_count, core_coords, out_fmt, window=0, global_cache=False, mask_bits=0
):
    cores = [ttnn.CoreCoord(x, y) for x, y in core_coords]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    runtime = [
        (core, [tile_count * i // len(cores), tile_count * (i + 1) // len(cores)]) for i, core in enumerate(cores)
    ]
    q, idx, table, *rest = accessors
    y = rest[-1]
    mask = rest[0] if len(rest) == 2 else ()
    defines = (
        [
            ("ROWVM_POSITION", str(window)),
            ("ROWVM_GLOBAL", str(int(global_cache))),
            ("ROWVM_MASK_BITS", str(mask_bits) + "u"),
        ]
        if window
        else []
    )
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=True,
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    unpack[2] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode = unpack
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "relative_io.cpp"),
            core_ranges=grid,
            runtime_args=runtime,
            compile_time_args=[index_page, *q, *idx, *table, *mask],
            config=ttnn.ReaderConfigDescriptor(),
            defines=defines,
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "relative_io.cpp"),
            core_ranges=grid,
            runtime_args=runtime,
            compile_time_args=[1024 * out_fmt, *y],
            defines=[("ROWVM_WRITE", "1")],
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "relative_compute.cpp"),
            core_ranges=grid,
            compile_time_args=[out_fmt],
            runtime_args=runtime,
            config=compute,
        ),
    ]
    cbs = [_cb(grid, i, fmt) for i, fmt in ((0, 2), (1, 2), (2, 4), (3, 2), (16, out_fmt))]
    cbs.append(_cb(grid, 4, 4, page_size=128))
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernels


def relative_bias(x, indices, table, mask, *, dtype=BF, out=None, memory_config=None):
    """Gather pre-scaled BF16 table rows, dot with R heads, and add an FP32 row mask.

    Indices are an interleaved uint32 row [1,L] with L a multiple of 32.
    Every index must be in bounds in the interleaved BF16 row-major table [N,32].
    """
    _validate(x, (1, 1, 24, 32))
    if x.dtype != BF:
        raise ValueError("rowvm relative bias requires BF16 relative heads")
    for tensor, expected_dtype in ((indices, ttnn.uint32), (table, BF)):
        if (
            len(tensor.shape) != 2
            or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
            or tensor.dtype != expected_dtype
            or tensor.is_sharded()
            or not tensor.is_allocated()
            or tensor.device() != x.device()
        ):
            raise ValueError("rowvm relative bias requires interleaved device row-major indices and table")
    width = indices.shape[-1]
    if indices.shape[0] != 1 or width == 0 or width % 32 or table.shape[-1] != 32:
        raise ValueError("rowvm relative bias requires indices [1,L] and table [N,32], with L a multiple of 32")
    _validate(mask, (1, 1, 1, width))
    if mask.dtype != FP:
        raise ValueError("rowvm relative bias requires an FP32 mask")
    return _relative_launch(
        x, [x, indices, table, mask], width, indices.padded_shape[-1] * 4, dtype, out, memory_config
    )


def relative_bias_from_position(x, position, table, width, *, window, global_cache, masked, out):
    """Derive the learned relative-table index and mask from the absolute position."""
    _validate(x, (1, 1, 24, 32))
    if (
        x.dtype != BF
        or position.dtype != ttnn.int32
        or tuple(position.shape) != (1,)
        or position.layout != ttnn.ROW_MAJOR_LAYOUT
        or position.is_sharded()
        or not position.is_allocated()
        or table.dtype != BF
        or table.layout != ttnn.ROW_MAJOR_LAYOUT
        or table.is_sharded()
        or not table.is_allocated()
        or tuple(table.shape) != (window + 1, 32)
        or window <= 0
        or window & (window - 1)
        or width <= 0
        or width % 32
        or (not global_cache and width != window)
    ):
        raise ValueError("invalid relative-bias window, table or position")
    return _relative_launch(x, [x, position, table], width, 4, BF, out, None, window, global_cache, _bits(masked))


def _relative_launch(
    x, inputs, width, index_page, dtype, out, memory_config, window=0, global_cache=False, mask_bits=0
):
    y = _output(x, (1, 1, 24, width), dtype, out, memory_config)
    dev = x.device()
    if any(t.device() != dev for t in [*inputs, y]) or y.buffer_unique_id() in {t.buffer_unique_id() for t in inputs}:
        raise ValueError("rowvm relative bias requires independent output storage on the same device as the inputs")
    grid = dev.compute_with_storage_grid_size()
    tiles = width // 32
    coords = tuple((i % grid.x, i // grid.x) for i in range(min(tiles, grid.x * grid.y)))
    accessors = tuple(tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in [*inputs, y])
    with _launch_lock:
        desc, kernels = _relative_program(
            accessors, index_page, tiles, coords, 4 if dtype == FP else 2, window, global_cache, mask_bits
        )
        kernels[0].common_runtime_args = [t.buffer_address() for t in inputs]
        kernels[1].common_runtime_args = [y.buffer_address()]
        desc.kernels = kernels
        ttnn.generic_op([*inputs, y], desc)
    return y


def _bits(value):
    return unpack("<I", pack("<f", value))[0]


def _width(x, *, allow_sharded=False):
    _validate(x, allow_sharded=allow_sharded)
    width = x.shape[-1]
    if tuple(x.shape)[:3] != (1, 1, 1) or width not in range(16, 4609, 16):
        raise ValueError("rowvm requires a [1,1,1,N] row aligned to 16 columns")
    return width


def _stripes(width, limit=256):
    chunk = min(width, limit)
    return chunk, (width + chunk - 1) // chunk


def _stripe(tensor, chunk, width, **kwargs):
    return _Region(tensor, chunk, col_step=chunk, extent=width, **kwargs)


def _elementwise(op, x, other, *, dtype, out, memory_config):
    width = _width(x)
    _validate(other, x.shape)
    chunk, tasks = _stripes(width)
    y = _output(x, x.shape, dtype, out, memory_config)
    args = (op, chunk, 2 if dtype == BF else 0)
    return _run(args, [_stripe(t, chunk, width) for t in (x, other)], [_stripe(y, chunk, width)], tasks)


def add(x, other, *, dtype=BF, out=None, memory_config=None):
    """Add two rows, with one optional BF16 output rounding."""
    return _elementwise(_ADD, x, other, dtype=dtype, out=out, memory_config=memory_config)


def sigmoid_gate(x, logits, *, dtype=BF, out=None, memory_config=None):
    """Multiply a row by accurate sigmoid of the gate logits."""
    return _elementwise(_GATE, x, logits, dtype=dtype, out=out, memory_config=memory_config)


def depthwise_conv(history, x, taps, *, dyn=None, dtype=FP, out=None, memory_config=None):
    """Shift FP32 history in place and apply four pre-scaled taps."""
    width = _width(x)
    _validate(history, (1, 1, 4, width))
    _validate(taps, history.shape)
    if history.dtype != FP:
        raise ValueError("rowvm convolution history must use float32")
    if history.buffer_unique_id() in (x.buffer_unique_id(), taps.buffer_unique_id()):
        raise ValueError("rowvm convolution history cannot alias input or taps")
    limit = 128
    if dyn is not None:
        _validate(dyn, (1, 1, 1, 32))
        if width != 1024:
            raise ValueError("dynamic convolution requires eight heads of width 128")
        limit = 64
    chunk, tasks = _stripes(width, limit)
    readers = [
        _stripe(x, chunk, width),
        _stripe(history, chunk, width, rows=3),
        _stripe(taps, chunk, width, rows=4),
    ]
    if dyn is not None:
        readers.append(_Region(dyn, 32))
    y = _output(x, x.shape, dtype, out, memory_config)
    return _run(
        (_CONV, chunk, (2 if dtype == BF else 0) | (4 if dyn is not None else 0)),
        readers,
        [_stripe(y, chunk, width), _stripe(history, chunk, width, rows=4)],
        tasks,
    )


def conv_residual(history, x, taps, residual):
    """Shift convolution history, round the convolution, and add the residual in BF16."""
    width = _width(x)
    _validate(residual, x.shape)
    _validate(history, (1, 1, 4, width))
    _validate(taps, history.shape)
    if history.dtype != FP:
        raise ValueError("rowvm convolution history must use float32")
    chunk, stripes = _stripes(width, 128)
    y = _output(x, x.shape, BF, None, None)
    return _run(
        (_CONV_ADD, chunk, 2 | 16),
        [_stripe(t, chunk, width, rows=n) for t, n in ((x, 1), (history, 3), (taps, 4), (residual, 1))],
        [_stripe(y, chunk, width), _stripe(history, chunk, width, rows=4)],
        stripes,
    )


def _experts(x):
    _validate(x)
    if tuple(x.shape)[1:3] not in ((1, 8), (8, 1)):
        raise ValueError("rowvm experts require [1,1,8,W] or [1,8,1,W]")
    return tuple(x.shape)


def moe_hidden(gu, *, dtype=BF, out=None, memory_config=None):
    """Apply raw-gate sigmoid and the two softcaps to eight expert rows."""
    shape = _experts(gu)
    if shape[-1] not in (1152, 2304):
        raise ValueError("rowvm MoE activation requires gate | up rows of width 1152 or 2304")
    half = shape[-1] // 2
    chunk = 192
    y = _output(gu, shape[:-1] + (half,), dtype, out, memory_config)
    # Each task owns a stripe within one expert.
    return _run(
        (_MOE_HIDDEN, chunk, 2 if dtype == BF else 0),
        [_Region(gu, chunk, col=c, row_step=1, col_step=chunk) for c in (0, half)],
        [_Region(y, chunk, row_step=1, col_step=chunk)],
        8 * (half // chunk),
        task_cols=half // chunk,
    )


def moe_combine(experts, gamma, *, dtype=BF, out=None, memory_config=None):
    """Accumulate eight expert outputs with the router's unbiased weights."""
    shape = _experts(experts)
    width = shape[-1]
    if width not in range(32, 4609, 32):
        raise ValueError("rowvm MoE combine requires a width that is a multiple of 32 up to 4608")
    _validate(gamma, (1, 1, 1, 8))
    chunk, tasks = _stripes(width, 128)
    y = _output(experts, (1, 1, 1, width), dtype, out, memory_config)
    return _run(
        (_MOE_COMBINE, chunk, 2 if dtype == BF else 0),
        [_stripe(experts, chunk, width, rows=8), _Region(gamma, 16)],
        [_stripe(y, chunk, width)],
        tasks,
    )


def message_input(u, m, *, normalized_u=None, dtype=BF, out=None, memory_config=None, _group=None):
    """Build the BF16 [RMSNorm(u) | RMSNorm(m)] controller input."""
    width = _width(u)
    _validate(m, u.shape)
    if normalized_u is not None:
        _validate(normalized_u, u.shape)
    y = _output(u, (1, 1, 1, 2 * width), dtype, out, memory_config, allow_sharded=True)
    first = normalized_u if normalized_u is not None else u
    flags = (2 if dtype == BF else 0) | (4 if normalized_u is not None else 0)
    if width >= 1024:
        chunk, stripes = _stripes(width)
        return _run(
            (_GATE_INPUT_STRIPED, chunk, flags, 0, width, stripes),
            [_stripe(t, chunk, width) for t in (first, m)],
            [_stripe(y, chunk, width, group_col_step=width)],
            2 * stripes,
            task_cols=stripes,
            group=_group,
        )
    return _run(
        (_GATE_INPUT, width, flags),
        [_Region(first, width), _Region(m, width)],
        [_Region(y, width, rows=2, concat=True)],
        group=_group,
    )


def message_output(u, m, forget, gate, *, weight=None, dtype=FP, out=None, memory_config=None):
    """Blend raw controller logits. A weight also enables final RMSNorm."""
    width = _width(u)
    for tensor in (m, forget, gate):
        _validate(tensor, u.shape)
    if weight is not None:
        _validate(weight, u.shape)
    tensors = (u, m, forget, gate) if weight is None else (u, m, forget, gate, weight)
    y = _output(u, u.shape, dtype, out, memory_config)
    flags = (1 if weight is not None else 0) | (2 if dtype == BF else 0)
    if weight is not None and width < 1024:
        return _run((_MESSAGE, width, flags), [_Region(t, width) for t in tensors], [_Region(y, width)])
    chunk, tasks = _stripes(width)
    args = (_MESSAGE, chunk, flags) if weight is None else (_MESSAGE_STRIPED, chunk, flags, 0, width, tasks)
    return _run(args, [_stripe(t, chunk, width) for t in tensors], [_stripe(y, chunk, width)], tasks)


def message_project(u, m, h, forget, gate, weight, *, out=None, memory_config=None, _group=None):
    """Project the BF16-rounded controller row and emit the weighted BF16 blend."""
    _validate(u, (1, 1, 1, 2304))
    _validate(m, u.shape)
    _validate(h, (1, 1, 1, 64), allow_sharded=True)
    _validate(weight, u.shape)
    for t in (forget, gate):
        _validate(t, (1, 1, 64, 2304))
        if t.dtype != FP:
            raise ValueError("message projection weights must use FP32 tiles")
    chunk, tasks = _stripes(2304, 64)
    y = _output(u, u.shape, BF, out, memory_config, allow_sharded=True)
    readers = [
        *(_stripe(t, chunk, 2304) for t in (u, m, weight)),
        _Region(h, 64, whole_tiles=True),
        *(_Region(t, chunk, rows=2, col_step=chunk, whole_tiles=True) for t in (forget, gate)),
    ]
    return _run(
        (_MESSAGE_UP, chunk, 3, 0, 2304, tasks),
        readers,
        [_stripe(y, chunk, 2304)],
        tasks,
        group=_group,
    )


def qkv_message_inputs(operands):
    """Prepare the three independent controller inputs in one dispatch."""
    group = _Group()
    rows = tuple(message_input(u, m, memory_config=mem, _group=group) for u, m, mem in operands)
    group.launch()
    return rows


def qkv_message_projections(operands):
    """Run the three independent controller epilogues in one dispatch."""
    group = _Group()
    rows = tuple(message_project(*args, memory_config=mem, _group=group) for args, mem in operands)
    group.launch()
    return rows


def attnres_mix(
    bank,
    partial,
    queries,
    valid_rows,
    *,
    bank_start=0,
    weight=None,
    dtype=FP,
    out=None,
    memory_config=None,
    convolution=None,
    shared_prefix=False,
    append=None,
    round_residual=False,
):
    """Read a normalized bank with up to four pre-scaled queries.

    Queries and optional per-query RMS weights have shape [1,1,nq,N].
    Returns a tuple of independently allocated [1,1,1,N] rows.
    With a convolution, partial is the branch input and the convolution
    tuple supplies its history, taps, optional residual, and output row.
    """
    width = _width(partial, allow_sharded=convolution is not None)
    _validate(bank, (1, 1, 32, width))
    _validate(queries)
    nq = queries.shape[2]
    _validate(queries, (1, 1, nq, width))
    if (
        bank.dtype != FP
        or nq not in range(1, 5)
        or not 0 <= bank_start <= 32
        or not 0 <= valid_rows <= 32
        or bank_start + max(0, valid_rows - shared_prefix) > 32
    ):
        raise ValueError("rowvm mix requires an FP32 bank, 0–32 valid rows and 1–4 queries")
    if width % 32:
        raise ValueError("rowvm mix width must be a multiple of 32")
    if weight is not None:
        _validate(weight, queries.shape)
    if out is not None and len(out) != nq:
        raise ValueError("rowvm mix needs one output tensor per query")
    ys = tuple(
        _output(partial, partial.shape, dtype, None if out is None else out[q], memory_config) for q in range(nq)
    )
    if any(t.memory_config() != ys[0].memory_config() for t in ys):
        raise ValueError("rowvm mix outputs must use the same memory configuration")
    if valid_rows == 0 and nq == 1 and weight is None and convolution is None:
        return (rms_norm(partial, dtype=dtype, out=ys[0]),)
    chunk, stripes = _stripes(width, _MIX_CHUNK)
    readers = [
        _stripe(partial, chunk, width),
        _stripe(queries, chunk, width, row_step=1),
        _stripe(
            bank,
            chunk,
            width,
            rows=max(1, valid_rows),
            row=bank_start if valid_rows else 0,
            shared_prefix=shared_prefix and valid_rows > 0,
        ),
    ]
    if weight is not None:
        readers.append(_stripe(weight, chunk, width, row_step=1))
    flags = (1 if weight is not None else 0) | (2 if dtype == BF else 0)
    op = _MIX
    writers = [_stripe(ys[0], chunk, width)]
    if convolution is not None:
        if nq != 1:
            raise ValueError("a convolution feeding a mix requires a single query")
        history, taps, residual, z = convolution
        _validate(history, (1, 1, 4, width))
        _validate(taps, history.shape)
        _validate(z, partial.shape)
        if history.dtype != FP or z.dtype != (BF if residual is None or round_residual else FP):
            raise ValueError("invalid convolution history or residual output precision")
        if residual is not None:
            _validate(residual, partial.shape)
            flags |= 16
        if round_residual:
            flags |= 64
        if weight is None:
            # Keep convolution input CBs at the same offsets for both variants.
            readers.append(_stripe(partial, chunk, width))
        readers.extend(_stripe(t, chunk, width, rows=n) for t, n in ((partial, 1), (history, 3), (taps, 4)))
        if residual is not None:
            readers.append(_stripe(residual, chunk, width))
        writers.extend((_stripe(history, chunk, width, rows=4), _stripe(z, chunk, width)))
        op = _CONV_MIX
    if append is not None:
        entry, row = append
        if op != _CONV_MIX or residual is not None or valid_rows >= 32 or not 0 <= row < 32:
            raise ValueError("bank append requires a block-boundary convolution feeding the mix")
        _validate(entry, partial.shape)
        readers.append(_stripe(entry, chunk, width))
        writers.append(_stripe(bank, chunk, width, row=row))
        flags |= 32
        valid_rows += 1
    _run(
        (op, chunk, flags, valid_rows, width, stripes),
        readers,
        writers,
        nq * stripes,
        task_cols=stripes,
        query_outputs=ys,
    )
    return ys


def router(logits, bias, *, ids_out=None, gamma_out=None, memory_config=None):
    """Return sorted uint16 sparse-matmul IDs and FP32 unbiased gamma."""
    _validate(logits, (1, 1, 1, 128))
    _validate(bias, logits.shape)
    shape = (1, 1, 1, 8)
    mem = memory_config or logits.memory_config()
    gamma = _output(logits, shape, FP, gamma_out, memory_config)
    if ids_out is None:
        ids_out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape),
            ttnn.uint16,
            ttnn.ROW_MAJOR_LAYOUT,
            logits.device(),
            mem,
        )
    if (
        tuple(ids_out.shape) != shape
        or ids_out.layout != ttnn.ROW_MAJOR_LAYOUT
        or ids_out.dtype != ttnn.uint16
        or ids_out.is_sharded()
        or not ids_out.is_allocated()
        or (memory_config is not None and ids_out.memory_config() != memory_config)
    ):
        raise ValueError("router IDs require an interleaved [1,1,1,8] uint16 row-major device tensor")
    _run((_ROUTER, 128, 0), [_Region(logits, 128), _Region(bias, 128)], [_Region(gamma, 16)], indices=ids_out)
    return ids_out, gamma


@lru_cache(maxsize=4)
def _head_tail_program(core_coords, grid_width, shard_tiles, total_tiles, vocab, accessors):
    grid = _rects(core_coords)
    args = [shard_tiles, total_tiles, vocab, len(core_coords), grid_width, *accessors[0], *accessors[1]]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "head_tail.cpp"),
            core_ranges=grid,
            compile_time_args=args,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "head_tail.cpp"),
            core_ranges=grid,
            compile_time_args=args,
            defines=[("HEAD_TAIL_SECOND", "1")],
            config=ttnn.ReaderConfigDescriptor(),
        ),
    ]
    cbs = [_cb(grid, 0, 2, page_size=shard_tiles * 64 + 64), _cb(grid, 1, 4, page_size=32 * len(core_coords))]
    semaphores = [ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, grid, 0)]
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernels


def head_tail(logits, out, index, vocab):
    """Copy width-sharded BF16 head logits into a row-major row and write the masked argmax.

    Ties resolve to the lowest index; columns at or beyond vocab are excluded.
    """
    shard = logits.memory_config().shard_spec
    width = logits.shape[-1]
    if (
        logits.dtype != BF
        or logits.layout != TILE
        or tuple(logits.shape)[:3] != (1, 1, 1)
        or logits.memory_config().memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED
        or logits.memory_config().buffer_type != ttnn.BufferType.L1
        or shard is None
        or shard.orientation != ttnn.ShardOrientation.ROW_MAJOR
        or next(iter(shard.shape)) != 32
    ):
        raise ValueError("head_tail requires row-major width-sharded BF16 [1,1,1,N] L1 logits")
    if (
        out.layout != ttnn.ROW_MAJOR_LAYOUT
        or out.dtype != BF
        or tuple(out.shape) != tuple(logits.shape)
        or out.is_sharded()
        or out.memory_config().buffer_type != ttnn.BufferType.DRAM
    ):
        raise ValueError("head_tail writes an interleaved row-major BF16 DRAM row")
    if index.dtype != ttnn.uint32 or index.layout != ttnn.ROW_MAJOR_LAYOUT or index.is_sharded():
        raise ValueError("head_tail writes an interleaved uint32 row-major index")
    shard_tiles = shard.shape[1] // 32
    total_tiles = width // 32
    workers = (total_tiles + shard_tiles - 1) // shard_tiles
    device = logits.device()
    grid = device.compute_with_storage_grid_size()
    if (
        width % 32
        or not 0 < vocab <= width
        or workers > grid.x * grid.y
        or shard.grid != ttnn.num_cores_to_corerangeset(workers, grid, row_wise=True)
    ):
        raise ValueError("head_tail requires the head's row-major shard grid")
    coords = tuple((i % grid.x, i // grid.x) for i in range(workers))
    accessors = tuple(tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in (out, index))
    leader = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
    with _launch_lock:
        desc, kernels = _head_tail_program(coords, grid.x, shard_tiles, total_tiles, vocab, accessors)
        for kernel in kernels:
            kernel.common_runtime_args = [
                logits.buffer_address(),
                out.buffer_address(),
                index.buffer_address(),
                leader.x | (leader.y << 16),
            ]
        desc.kernels = kernels
        ttnn.generic_op([logits, out, index], desc)
    return out, index


_ROUTER_STRIPES = int(os.environ.get("INK_ROUTER_STRIPES", "24"))


@lru_cache(maxsize=4)
def _router_program(accessors, stripes, grid_width):
    coords = tuple((i % grid_width, i // grid_width) for i in range(stripes))
    grid = _rects(coords)
    k_tiles = 72 // stripes
    args = [k_tiles, stripes, grid_width]
    io_args = [*args, *(a for acc in accessors for a in acc)]
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=True,
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for cb in (2, 3, 8):
        unpack[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode = unpack
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "router_io.cpp"),
            core_ranges=grid,
            compile_time_args=io_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "router_io.cpp"),
            core_ranges=grid,
            compile_time_args=io_args,
            defines=[("ROUTER_WRITER", "1")],
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "router_compute.cpp"), core_ranges=grid, compile_time_args=args, config=compute
        ),
    ]
    aggregate = (stripes * 512 + 4095) // 4096
    cbs = [
        _cb(grid, 0, 2, k_tiles),
        _cb(grid, 1, 2, 4 * k_tiles),
        _cb(grid, 2, 4),
        _cb(grid, 3, 4, aggregate),
        _cb(grid, 8, 4),
        _cb(grid, 16, 4, 4),
        _cb(grid, 17, 4),
        _cb(grid, 18, 4, 2),
    ]
    semaphores = [ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, grid, 0)]
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernels


def fused_router(x, weight, bias, *, ids_out=None, gamma_out=None):
    """Router projection, sigmoid + correction bias, top-8 and unbiased gamma in one launch.

    x is a BF16 TILE [1,1,1,2304] row, weight the interleaved BF16 TILE [2304,128] router matrix and
    bias a row-major FP32 [1,1,1,128] row. The projection accumulates BF16 products in FP32 DEST
    per K stripe and sums the stripe partials in FP32.
    """
    _validate(x, (1, 1, 1, 2304))
    if x.dtype != BF:
        raise ValueError("fused router requires a BF16 activation")
    if tuple(weight.shape)[-2:] != (2304, 128) or weight.dtype != BF or weight.layout != TILE or weight.is_sharded():
        raise ValueError("fused router requires an interleaved BF16 TILE [2304,128] weight")
    if bias.layout != ttnn.ROW_MAJOR_LAYOUT or bias.dtype != FP or tuple(bias.shape) != (1, 1, 1, 128):
        raise ValueError("fused router requires a row-major FP32 [1,1,1,128] bias")
    device = x.device()
    grid = device.compute_with_storage_grid_size()
    if grid.x * grid.y < _ROUTER_STRIPES:
        raise ValueError("fused router requires one worker per stripe")
    mem = x.memory_config()
    shape = (1, 1, 1, 8)
    gamma = _output(x, shape, FP, gamma_out, mem)
    if ids_out is None:
        ids_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, device, mem)
    operands = (x, weight, bias, ids_out, gamma)
    accessors = tuple(tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in operands)
    leader = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
    with _launch_lock:
        desc, kernels = _router_program(accessors, _ROUTER_STRIPES, grid.x)
        common = [t.buffer_address() for t in operands] + [leader.x | (leader.y << 16)]
        kernels[0].common_runtime_args = common
        kernels[1].common_runtime_args = common
        desc.kernels = kernels
        ttnn.generic_op([x, weight, bias, ids_out, gamma], desc)
    return ids_out, gamma


_MATVEC_STRIPES = int(os.environ.get("INK_MATVEC_STRIPES", "18"))


@lru_cache(maxsize=8)
def _matvec_program(accessors, k_tiles, stripes, groups, n_tiles, grid_width):
    coords = tuple((i % grid_width, i // grid_width) for i in range(stripes * groups))
    grid = _rects(coords)
    args = [k_tiles, stripes, groups, grid_width, n_tiles]
    io_args = [*args, *(a for acc in accessors for a in acc)]
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=True,
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    unpack[3] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode = unpack
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "matvec_io.cpp"),
            core_ranges=grid,
            compile_time_args=io_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "matvec_io.cpp"),
            core_ranges=grid,
            compile_time_args=io_args,
            defines=[("MATVEC_WRITER", "1")],
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "matvec_compute.cpp"), core_ranges=grid, compile_time_args=args, config=compute
        ),
    ]
    aggregate = (stripes * n_tiles * 128 + 4095) // 4096
    cbs = [
        _cb(grid, 0, 2, k_tiles),
        _cb(grid, 1, 2, n_tiles * k_tiles),
        _cb(grid, 3, 4, aggregate),
        _cb(grid, 16, 4, n_tiles),
        _cb(grid, 17, 2),
    ]
    semaphores = [ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, grid, 0)]
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernels


def matvec_group(pairs, *, outs=None):
    """Independent BF16 row projections y_g = round_bf16(x_g @ W_g) in one launch.

    Each x_g is a BF16 TILE [1,1,1,K] row and each W_g an interleaved BF16 TILE [K,N] matrix in DRAM.
    Products accumulate in FP32 DEST per K stripe; stripe partials add in FP32 before one
    round-to-nearest-even. Outputs are interleaved L1 BF16 TILE [1,1,1,N] rows.
    """
    pairs = tuple(pairs)
    k = pairs[0][0].shape[-1]
    n = pairs[0][1].shape[-1]
    stripes = _MATVEC_STRIPES
    if k % (32 * stripes) or n % 32:
        raise ValueError("matvec_group requires K divisible into stripes of whole tiles")
    device = pairs[0][0].device()
    grid = device.compute_with_storage_grid_size()
    if len(pairs) * stripes > grid.x * grid.y:
        raise ValueError("matvec_group requires one worker per stripe per projection")
    ys = []
    for g, (x, w) in enumerate(pairs):
        _validate(x, (1, 1, 1, k), allow_sharded=True)
        if x.dtype != BF or w.dtype != BF or w.layout != TILE or w.is_sharded() or tuple(w.shape)[-2:] != (k, n):
            raise ValueError("matvec_group requires BF16 rows and interleaved BF16 TILE weights")
        y = (
            outs[g]
            if outs is not None
            else ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, n)), BF, TILE, device, ttnn.L1_MEMORY_CONFIG)
        )
        ys.append(y)
    first = (pairs[0][0], pairs[0][1], ys[0])
    accessors = tuple(tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in first)
    for (x, w), y in zip(pairs, ys):
        if tuple(tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in (x, w, y)) != accessors:
            raise ValueError("matvec_group operands must share placements")
    leaders = []
    for g in range(len(pairs)):
        i = g * stripes
        core = device.worker_core_from_logical_core(ttnn.CoreCoord(i % grid.x, i // grid.x))
        leaders.append(core.x | (core.y << 16))
    with _launch_lock:
        desc, kernels = _matvec_program(accessors, k // 32 // stripes, stripes, len(pairs), n // 32, grid.x)
        common = [t.buffer_address() for (x, w), y in zip(pairs, ys) for t in (x, w, y)] + leaders
        kernels[0].common_runtime_args = common
        kernels[1].common_runtime_args = common
        desc.kernels = kernels
        ttnn.generic_op([t for x, w in pairs for t in (x, w)] + ys[::-1], desc)
    return tuple(ys)


@lru_cache(maxsize=4)
def _feedback_program(accessors, row_bytes, lmax):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "feedback.cpp"),
        core_ranges=grid,
        compile_time_args=[row_bytes, lmax, *(a for acc in accessors for a in acc)],
        config=ttnn.WriterConfigDescriptor(),
    )
    cbs = [_cb(grid, 0, 2, page_size=row_bytes + 8 * 64)]
    desc = ttnn.ProgramDescriptor(kernels=[kernel], cbs=cbs)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernel


def token_feedback(index, table, x, position, tau_table, tau, ids):
    """Write the next greedy step's inputs on device from the head's argmax.

    x <- table[index] (row-major BF16 row), position <- position + 1, tau <- tau_table[position],
    ids[position] <- index. The table rows must already hold the normalized embeddings.
    """
    row_bytes = x.shape[-1] * 2
    lmax = tau_table.shape[-1]
    if (
        table.layout != ttnn.ROW_MAJOR_LAYOUT
        or x.layout != ttnn.ROW_MAJOR_LAYOUT
        or table.dtype != BF
        or x.dtype != BF
        or table.shape[-1] != x.shape[-1]
        or tuple(ids.shape)[-1] != lmax
        or tau_table.dtype != FP
        or ids.dtype != ttnn.uint32
        or row_bytes % 64
    ):
        raise ValueError("token feedback requires row-major BF16 table/row, FP32 tau table and uint32 IDs")
    operands = (index, table, x, position, tau_table, tau, ids)
    accessors = tuple(tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in operands)
    with _launch_lock:
        desc, kernel = _feedback_program(accessors, row_bytes, lmax)
        kernel.common_runtime_args = [t.buffer_address() for t in operands]
        desc.kernels = [kernel]
        ttnn.generic_op([index, table, tau_table, x, position, tau, ids], desc)


_MESSAGE_DOWN_STRIPES = 18


@lru_cache(maxsize=8)
def _message_down_program(accessors, groups, grid_width):
    coords = tuple((i % grid_width, i // grid_width) for i in range(_MESSAGE_DOWN_STRIPES * groups))
    grid = _rects(coords)
    io_args = [9, groups, grid_width, *(a for acc in accessors for a in acc)]
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=True,
    )
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    for cb in (3, 4, 8, 9):
        unpack[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute.unpack_to_dest_mode = unpack
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "msgdown_io.cpp"),
            core_ranges=grid,
            compile_time_args=io_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "msgdown_io.cpp"),
            core_ranges=grid,
            compile_time_args=io_args,
            defines=[("MSGDOWN_WRITER", "1")],
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "msgdown_compute.cpp"),
            core_ranges=grid,
            compile_time_args=[9, groups, grid_width, 2304],
            config=compute,
        ),
    ]
    x_tiles = ttnn.CBDescriptor(
        total_size=512 * 8,
        core_ranges=grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0, data_format=BF, page_size=512, tile=ttnn.TileDescriptor(ttnn.Tile([8, 32]))
            )
        ],
    )
    cbs = [
        x_tiles,
        _cb(grid, 1, 2, 16),
        _cb(grid, 3, 4, 2),
        _cb(grid, 4, 4),
        _cb(grid, 8, 4),
        _cb(grid, 9, 4),
        _cb(grid, 17, 4, 2),
        _cb(grid, 18, 4),
        _cb(grid, 19, 4),
        _cb(grid, 20, 2),
        _cb(grid, 21, 2),
    ]
    semaphores = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, grid, 0) for i in range(3)]
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernels


def message_down(operands):
    """BF16 h = round(RMSNorm(u) | RMSNorm(m) as BF16 @ W) for independent gates in one launch.

    Each operand is (u, m, W) with FP32 TILE [1,1,1,2304] rows and an interleaved BF16 TILE [4608,64]
    matrix. Each half normalizes over its 2304 columns; projection partials add in FP32 before one
    round-to-nearest-even. Returns interleaved L1 BF16 TILE [1,1,1,64] rows.
    """
    operands = tuple(operands)
    device = operands[0][0].device()
    grid = device.compute_with_storage_grid_size()
    workers = _MESSAGE_DOWN_STRIPES * len(operands)
    if workers > grid.x * grid.y:
        raise ValueError("message_down requires one worker per stripe per projection")
    hs = []
    for u, m, w in operands:
        for t in (u, m):
            _validate(t, (1, 1, 1, 2304))
            if t.dtype != FP or t.layout != TILE or t.is_sharded():
                raise ValueError("message_down requires interleaved FP32 TILE rows")
        if w.dtype != BF or w.layout != TILE or w.is_sharded() or tuple(w.shape)[-2:] != (4608, 64):
            raise ValueError("message_down requires an interleaved BF16 TILE [4608,64] weight")
        hs.append(ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, 64)), BF, TILE, device, ttnn.L1_MEMORY_CONFIG))
    spec = lambda t: tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    accessors = (spec(operands[0][0]), spec(operands[0][1]), spec(operands[0][2]), spec(hs[0]))
    for (u, m, w), h in zip(operands, hs):
        if (spec(u), spec(m), spec(w), spec(h)) != accessors:
            raise ValueError("message_down operands must share placements")
    coords = [device.worker_core_from_logical_core(ttnn.CoreCoord(i % grid.x, i // grid.x)) for i in range(workers)]
    with _launch_lock:
        desc, kernels = _message_down_program(accessors, len(operands), grid.x)
        common = [t.buffer_address() for (u, m, w), h in zip(operands, hs) for t in (u, m, w, h)]
        common += [c.x | (c.y << 16) for c in coords]
        kernels[0].common_runtime_args = common
        kernels[1].common_runtime_args = common
        desc.kernels = kernels
        ttnn.generic_op([t for u, m, w in operands for t in (u, m, w)] + hs[::-1], desc)
    return tuple(hs)
