"""Fixed-shape chunked prefill sharing InklingTT2 weights, KV, and convolution histories."""

import math
import os
import struct
from functools import lru_cache
from pathlib import Path
from threading import Lock

import torch
import ttnn

from inkling_tt import dense, experts_stream
from inkling_tt.model import (
    DR,
    EPS,
    HD,
    HKV,
    HQ,
    LOG_ALPHA,
    LOG_FLOOR,
    LOOP0,
    LOOP1,
    NPASS,
    TOPK,
    VOCAB,
    D,
    E,
    F,
    W,
    visit_slot,
)
from inkling_tt.model2 import BF, DRAM, FP, RM, TILE, act, bf16, f32

ISQ = 1.0 / math.sqrt(HD)
_ROUTER_K_BLOCK = 96
_KERNELS = Path(__file__).resolve().parent / "kernels" / "prefill"
_LOCK = Lock()


def _tile_cb(grid, index, dtype, pages=1, size=None):
    size = size or (2048 if dtype == BF else 4096)
    return ttnn.CBDescriptor(
        total_size=size * pages,
        core_ranges=grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size),
        ],
    )


def _tasks(device, count):
    g = device.compute_with_storage_grid_size()
    workers = min(g.x * g.y, count)
    return tuple((i % g.x, i // g.x, i * count // workers, (i + 1) * count // workers) for i in range(workers))


def _kv_projection_program(grid, chunk):
    # TTNN's interleaved 2D matmul uses K_tiles / grid.x when the
    # grid divides K_tiles. Keep single-tile K blocks on that grid too
    # so K/V accumulation matches the 11- and 13-column paths.
    if (D // 32) % grid.x:
        return None
    m = (chunk // 32 + grid.y - 1) // grid.y
    n = (HKV * HD // 32 + grid.x - 1) // grid.x
    subblock_h = next((h for h in (4, 3) if m % h == 0), 1)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=1,
        out_subblock_h=subblock_h,
        out_subblock_w=n if subblock_h == 1 else 1,
        out_block_h=m,
        out_block_w=n,
        per_core_M=m,
        per_core_N=n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def _launch(factory, args, *bindings, **options):
    tensors = list({id(t): t for group in bindings for t in group}.values())
    with _LOCK:
        desc, kernels = factory(
            *args,
            tuple(
                tuple(tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in group) for group in bindings
            ),
            **options,
        )
        for k, group in zip(kernels, bindings):
            k.common_runtime_args = [t.buffer_address() for t in group]
        desc.kernels = kernels
        ttnn.generic_op(tensors, desc)


def _grid(tasks):
    rt = [(ttnn.CoreCoord(x, y), [start, end]) for x, y, start, end in tasks]
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, _ in rt]), rt


def _compute(fp32_inputs, full_sync=False):
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=full_sync,
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for cb in fp32_inputs:
        modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    return cfg


def _program(kernels, buffers):
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=buffers)
    desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
    return desc, kernels


@lru_cache(maxsize=32)
def _attention_program(tasks, query_tiles, cache_tiles, is_global, retain_scores, specs, project=False):
    grid, rt = _grid(tasks)
    args = [query_tiles, cache_tiles, int(is_global), HQ, int(retain_scores)]
    cbs = [
        _tile_cb(grid, i, dt, pages, size)
        for i, dt, pages, size in (
            (0, BF, 4, None),
            (1, BF, 8, None),
            (2, BF, 32, None),
            (3, FP, 2, None),
            (4, BF, 1, None),
            (5, FP, 32, None),
            (6, ttnn.uint32, 1, 32),
            (7, ttnn.uint32, 1, 32),
            (8, FP, 2, None),
            (9, FP, 1, None),
            (10, FP, 1, None),
            (11, BF, 8, None),
            (12, FP, 4, None),
            (14, FP, 2, None),
            (15, ttnn.uint32, 1, 32),
            (16, BF, 8, None),
            (17, ttnn.uint32, 1, 32),
        )
    ]
    if project:
        cbs += [_tile_cb(grid, 18, BF, 2), _tile_cb(grid, 19, BF, 32)]
    defines = [("PREFILL_DISTANCE", "1")] if project else []
    if retain_scores:
        cbs.append(_tile_cb(grid, 13, FP, retain_scores))
    fp32_inputs = (3, 8, 9, 10, 12, 14) + ((13,) if retain_scores else ())
    reader, writer = tuple(sum(group, ()) for group in specs)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "attention_io.cpp"),
            core_ranges=grid,
            compile_time_args=[*args, *reader],
            runtime_args=rt,
            defines=defines,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "attention_io.cpp"),
            core_ranges=grid,
            compile_time_args=[*args, *writer],
            runtime_args=rt,
            defines=[("PREFILL_WRITE", "1"), *defines],
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "attention_compute.cpp"),
            core_ranges=grid,
            compile_time_args=args,
            runtime_args=rt,
            defines=defines,
            config=_compute(fp32_inputs),
        ),
    ]
    return _program(kernels, cbs)


def _tiled_attention(
    q,
    k,
    v,
    cache_k,
    cache_v,
    sink,
    distance,
    meta,
    is_global,
    out,
    *,
    retain_scores=False,
    global_score_tiles=0,
    table=None,
):
    """Two-pass biased attention. With `table`, `distance` is the padded relative query
    [1, HQ, T, 32] and the kernel forms the distance logits itself."""
    tiles = q.shape[-2] // 32
    inputs = (meta, q, k, v, cache_k, cache_v, sink, distance) + ((table,) if table is not None else ())
    _launch(
        _attention_program,
        (
            _tasks(q.device(), HQ * tiles),
            tiles,
            cache_k.shape[-2] // 32,
            is_global,
            global_score_tiles if is_global else (W // 32 + 1 if retain_scores else 0),
        ),
        inputs,
        (meta, out),
        project=table is not None,
    )
    return out


@lru_cache(maxsize=16)
def _state_program(tasks, tiles, capacity, cols, last, specs):
    grid, rt = _grid(tasks)
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "state.cpp"),
        core_ranges=grid,
        runtime_args=rt,
        compile_time_args=[tiles, capacity, cols, *sum(specs[0], ())],
        defines=[("PREFILL_LAST", "1")] if last else [],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [_tile_cb(grid, 0, BF, 2)])


@lru_cache(maxsize=16)
def _conv_program(tasks, cols, dynamic, tiles, x_dtype, specs):
    grid, rt = _grid(tasks)
    args = [cols, int(dynamic), tiles, int(x_dtype == BF)]
    reader, writer = tuple(sum(group, ()) for group in specs)
    cbs = [
        _tile_cb(grid, i, dt, pages, size)
        for i, dt, pages, size in (
            (0, x_dtype, 8, None),
            (1, FP, 4, None),
            (3, FP, 5, None),
            (4, ttnn.uint32, 2, 32),
            (16, BF, 2, None),
        )
    ]
    if dynamic:
        cbs.append(_tile_cb(grid, 2, FP, 8))
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "conv_io.cpp"),
            core_ranges=grid,
            compile_time_args=[*args, *reader],
            runtime_args=rt,
            defines=[("PREFILL_DYN", "1")] if dynamic else [],
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "conv_io.cpp"),
            core_ranges=grid,
            compile_time_args=[*args, *writer],
            runtime_args=rt,
            defines=[("PREFILL_WRITE", "1")],
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "conv_compute.cpp"),
            core_ranges=grid,
            compile_time_args=args,
            runtime_args=rt,
            config=_compute((1, 2) if x_dtype == BF else (0, 1, 2)),
        ),
    ]
    return _program(kernels, cbs)


@lru_cache(maxsize=32)
def _mix_program(tasks, cols, rows, depth, queries, specs):
    grid, rt = _grid(tasks)
    args = [cols, rows, depth, queries]
    reader, writer = tuple(sum(group, ()) for group in specs)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "mix_io.cpp"),
            core_ranges=grid,
            compile_time_args=[*args, *reader],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "mix_io.cpp"),
            core_ranges=grid,
            compile_time_args=[*args, *writer],
            runtime_args=rt,
            defines=[("PREFILL_WRITE", "1")],
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "mix_compute.cpp"),
            core_ranges=grid,
            compile_time_args=args,
            runtime_args=rt,
            config=_compute((0, 1)),
        ),
    ]
    return _program(
        kernels,
        [
            _tile_cb(grid, i, FP, n)
            for i, n in ((0, 2 * depth), (1, 2 * queries * depth), (2, depth), (16, 2 * queries))
        ],
    )


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


@lru_cache(maxsize=32)
def _heads_program(tasks, src_cols, col_offset, row_tiles, scaled, rms, threshold, specs):
    grid, rt = _grid(tasks)
    args = [src_cols, col_offset, row_tiles, int(scaled)]
    reader, writer = tuple(sum(group, ()) for group in specs)
    cbs = [_tile_cb(grid, 0, BF, 8), _tile_cb(grid, 16, BF, 8)]
    if scaled:
        cbs.append(_tile_cb(grid, 1, FP, 2))
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "heads_io.cpp"),
            core_ranges=grid,
            compile_time_args=[*args, *reader],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "heads_io.cpp"),
            core_ranges=grid,
            compile_time_args=[*args, *writer],
            runtime_args=rt,
            defines=[("PREFILL_WRITE", "1")],
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "heads_compute.cpp"),
            core_ranges=grid,
            compile_time_args=[*args, int(rms), _bits(threshold), _bits(EPS), _bits(1.0 / HD)],
            runtime_args=rt,
            config=_compute((1,), full_sync=True),
        ),
    ]
    return _program(kernels, cbs)


def _heads(x, col_offset, heads, *, rms=False, threshold=0.0, tau=None):
    """Clip or RMS-scale 128-channel heads per row and return them head-major."""
    t = x.shape[-2]
    out = _alloc(x.device(), (1, heads, t, HD))
    _launch(
        _heads_program,
        (
            _tasks(x.device(), heads * t // 32),
            x.shape[-1] // 32,
            col_offset // 32,
            t // 32,
            tau is not None,
            rms,
            threshold,
        ),
        (x, x if tau is None else tau),
        (out,),
    )
    return out


@lru_cache(maxsize=8)
def _gate_program(tasks, row_tiles, cols, specs):
    grid, rt = _grid(tasks)
    reader, writer = tuple(sum(group, ()) for group in specs)
    cbs = [_tile_cb(grid, 0, FP, 4), _tile_cb(grid, 1, FP, 4), _tile_cb(grid, 2, FP, 4), _tile_cb(grid, 16, FP, 2)]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "gate_io.cpp"),
            core_ranges=grid,
            compile_time_args=[row_tiles, cols, *reader],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "gate_io.cpp"),
            core_ranges=grid,
            compile_time_args=[row_tiles, cols, *writer],
            runtime_args=rt,
            defines=[("PREFILL_WRITE", "1")],
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "gate_compute.cpp"),
            core_ranges=grid,
            compile_time_args=[row_tiles],
            runtime_args=rt,
            config=_compute((2,)),
        ),
    ]
    return _program(kernels, cbs)


def _message_gate(h, forget, message, u, m):
    """u * (1 - tanh(h @ forget)) + m * tanh(h @ message) with FP32 controller weights."""
    rows, cols = u.shape[-2] // 32, u.shape[-1] // 32
    out = _alloc(u.device(), u.shape, FP)
    _launch(_gate_program, (_tasks(u.device(), rows * cols), rows, cols), (h, forget, message, u, m), (out,))
    return out


@lru_cache(maxsize=32)
def _copy_program(tasks, page, offset, specs):
    grid, rt = _grid(tasks)
    reader, writer = tuple(sum(group, ()) for group in specs)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "copy_io.cpp"),
            core_ranges=grid,
            compile_time_args=[page, offset, *reader],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "copy_io.cpp"),
            core_ranges=grid,
            compile_time_args=[page, offset, *writer],
            runtime_args=rt,
            defines=[("PREFILL_WRITE", "1")],
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    return _program(kernels, [_tile_cb(grid, 0, FP if page == 4096 else BF, 4)])


def _copy_into(src, dst, slot):
    """Write src into slot `slot` of dst along dim 1 (same tile shape and dtype)."""
    tiles = src.shape[-2] * src.shape[-1] // 1024
    page = 4096 if src.dtype == FP else 2048
    _launch(_copy_program, (_tasks(src.device(), tiles), page, slot * tiles), (src,), (dst,))


def _alloc(device, shape, dtype=BF, layout=TILE):
    return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, layout, device, DRAM)


def _stencil(x, taps, history, meta, dyn=None):
    out = _alloc(x.device(), x.shape)
    cols = x.shape[-1] // 32
    inputs = (meta, x, taps, history) + ((dyn,) if dyn is not None else ())
    _launch(
        _conv_program,
        (_tasks(x.device(), cols), cols, dyn is not None, x.shape[-2] // 32, x.dtype),
        inputs,
        (meta, out),
    )
    return out


class Prefill:
    """One sequence, one fixed chunk shape, shared with batch-one decode.

    Create every bucket before calling model.capture(). Each instance chains
    onto model.capture: decode is captured first, then each bucket's body and
    readout in creation order. Close buckets in reverse creation order.
    """

    def __init__(self, m, chunk=512, fixed=True, warmup=True, *, trace=True):
        if not 128 <= chunk <= 2048 or chunk % 32:
            raise ValueError("chunk must be a multiple of 32 between 128 and 2048")
        if m.lmax % 32:
            raise ValueError("lmax must be a multiple of 32")
        if not fixed:
            raise ValueError("prefill uses one fixed padded chunk shape")
        if m.trace_id is not None:
            raise ValueError("create Prefill before calling model.capture()")
        self.m, self.c, self.ops = m, m.c, m.ops
        self.chunk = chunk
        self.dev = m.c.dev
        grid = self.dev.compute_with_storage_grid_size()
        self._kv_program = _kv_projection_program(grid, chunk)
        self.body_trace = self.head_trace = None
        self._trace_enabled = trace
        self._message_dense = os.environ.get("INK_PREFILL_MESSAGE_DENSE", "1") == "1"
        self._broadcast_dense = os.environ.get("INK_PREFILL_DENSE", "broadcast") == "broadcast"
        self._retain_scores = os.environ.get("INK_PREFILL_KEEP_SCORES", "1") == "1"
        wide_score_tiles = 188 if grid.x * grid.y >= 130 else 160
        score_tiles = int(
            os.environ.get("INK_PREFILL_GLOBAL_SCORE_TILES", str(wide_score_tiles if chunk > 1024 else 97))
        )
        self._global_score_tiles = min(m.lmax // 32, max(0, score_tiles))
        self._warmed = False
        self.dev.enable_program_cache()
        self.rows = _alloc(self.dev, (1, 1, chunk, D))
        self.meta = _alloc(self.dev, (1, 8), ttnn.uint32, RM)
        self.tau = _alloc(self.dev, (1, chunk, 1, 1), FP)
        self.tau_col = _alloc(self.dev, (1, 1, chunk, 32), FP)
        self._fused_heads = os.environ.get("INK_PREFILL_HEADS", "1") == "1"
        self._fused_gate = os.environ.get("INK_PREFILL_GATE", "1") == "1"
        self._reuse_items = os.environ.get("INK_PREFILL_REUSE_ITEMS", "1") == "1"
        self._project_distance = os.environ.get("INK_PREFILL_PROJECT_DISTANCE", "1") == "1"
        self._bank_items = {}
        self._un16 = None
        self.attn_out = _alloc(self.dev, (1, 1, chunk, HQ * HD))
        self.last = _alloc(self.dev, (1, 1, 1, D))
        self.logits = _alloc(self.dev, (1, 1, 1, m.head.n))
        self.rel_weights = {}
        self.router_weights = {}
        for layer in m.layers:
            table = ttnn.to_torch(layer.rel_table)[:W].flip(0).T.contiguous()
            self.rel_weights[layer.i] = self.c.up(table.reshape(1, 1, 32, W))
            self.router_weights[layer.i] = ttnn.reshape(
                layer.router_w.tensor,
                [1, D // _ROUTER_K_BLOCK, _ROUTER_K_BLOCK, E],
            )

        stream = [layer.expert_stream is not None for layer in m.layers]
        broadcast_down = os.environ.get("INK_PREFILL_EXPERT_BROADCAST", str(int(chunk >= 512))) == "1"
        full_tiles = os.environ.get("INK_PREFILL_EXPERT_FULL_TILES", str(int(chunk >= 512))) == "1"
        default_lanes = min(7, experts_stream.prefill_lane_capacity(self.dev)) if full_tiles else 0
        lanes = int(os.environ.get("INK_PREFILL_EXPERT_LANES", default_lanes))
        wide = os.environ.get("INK_PREFILL_EXPERT_WIDE_L1")
        self.expert_workspace = (
            experts_stream.prepare_prefill(
                self.dev,
                chunk,
                broadcast_down=broadcast_down,
                full_tiles=full_tiles,
                lanes=lanes,
                wide_l1_blocks=None if wide is None else int(wide),
            )
            if any(stream)
            else None
        )
        if not all(stream):
            self.all_ids = self.c.up(torch.arange(E, dtype=torch.int32).reshape(1, 1, 1, E), ttnn.uint16, RM)
            self.stock_scores = self.c.up(torch.zeros(1, 1, chunk, E), FP, RM)
        self.ops.prefill_lin = self.lin

        if warmup:
            self._warm()
        self._decode_capture = m.capture
        m.capture = self._capture_with_decode

    def _warm(self):
        self._stage([0] * min(self.chunk, self.m.lmax), 0)
        self._body()
        self._readout()
        ttnn.synchronize_device(self.dev)
        self.m.reset()
        self._warmed = True

    def _capture_with_decode(self):
        if self.m.trace_id is None:
            self._release_traces()
            if not self._warmed:
                self._warm()
            self._decode_capture()
        if self._trace_enabled and self.body_trace is None:
            self.body_trace = self._capture(self._body)
            try:
                self.head_trace = self._capture(self._readout)
            except Exception:
                self._release_traces()
                raise

    def _capture(self, graph):
        trace = ttnn.begin_trace_capture(self.dev, cq_id=0)
        try:
            graph()
        except Exception:
            ttnn.end_trace_capture(self.dev, trace, cq_id=0)
            ttnn.release_trace(self.dev, trace)
            raise
        ttnn.end_trace_capture(self.dev, trace, cq_id=0)
        return trace

    def _release_traces(self):
        for trace in (self.body_trace, self.head_trace):
            if trace is not None:
                ttnn.release_trace(self.dev, trace)
        self.body_trace = self.head_trace = None

    def close(self):
        """Release captured prefill commands before closing the device."""
        self._release_traces()
        if getattr(self.m.capture, "__self__", None) is self:
            self.m.capture = self._decode_capture

    def lin(self, x, weight, dtype=BF):
        """Project all rows while preserving the DRAM-sharded decode accumulation order."""
        kw = dict(compute_kernel_config=weight.compute, dtype=dtype)
        if weight.input_shards is None:
            if self._kv_program is not None and (weight.k, weight.n) == (D, HKV * HD):
                kw["program_config"] = self._kv_program
            return ttnn.linear(x, weight.tensor, **kw)
        x = bf16(x)
        if self._message_dense and (weight.k, weight.n, weight.tensor.dtype) == (4608, 64, BF):
            y = dense.message_prefill_down(x, weight)
            return y if dtype == BF else ttnn.typecast(y, dtype)
        if self._broadcast_dense and weight.n in (2304, 6560) and weight.tensor.dtype == BF:
            return dense.linear_prefill_resident(x, weight, dtype)
        outputs = []
        for start in range(0, x.shape[-2], 32):
            block = ttnn.slice(x, [0, 0, start, 0], [1, 1, start + 32, weight.k])
            block = ttnn.to_memory_config(block, weight.input_shards)
            y = ttnn.linear(
                block, weight.tensor, program_config=weight.program, memory_config=weight.output_shards, **kw
            )
            outputs.append(ttnn.to_memory_config(y, DRAM))
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=-2)

    def _items(self, bank, current):
        """The depth-mixing operand [*bank, current], reusing one buffer per bank state."""
        key = tuple(id(entry) for entry in bank)
        cached = self._bank_items.get(key)
        if cached is not None and all(a is b for a, b in zip(cached[0], bank)):
            _copy_into(current, cached[1], len(bank))
            return cached[1]
        items = ttnn.concat([*bank, current], dim=1)
        if self._reuse_items:
            self._bank_items[key] = (list(bank), items)
            while len(self._bank_items) > 4:
                self._bank_items.pop(next(iter(self._bank_items)))
        return items

    def mix(self, bank, partial, qm, nq):
        current = self.ops.rms(partial)
        if not bank:
            return [current] * nq
        items = self._items(bank, current)
        scores = self.c.mm(items, qm[0])
        weights = ttnn.softmax(scores, dim=1)
        out = [_alloc(self.dev, (1, 1, self.chunk, D), FP) for _ in range(nq)]
        cols, rows = D // 32, self.chunk // 32
        _launch(
            _mix_program, (_tasks(self.dev, cols * rows), cols, rows, len(bank) + 1, nq), (items, weights), tuple(out)
        )
        return out

    @staticmethod
    def hclip(x, thr):
        x = f32(x)
        ss = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
        fac = [act("SQRT"), act("ADD_UNARY_SFPU", EPS), act("RECIP"), act("MUL_UNARY_SFPU", thr), act("MINIMUM", 1.0)]
        return ttnn.multiply(x, ss, input_tensor_b_activations=fac, dtype=BF)

    @staticmethod
    def hrms(x):
        x = f32(x)
        ss = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
        fac = [act("MUL_UNARY_SFPU", 1.0 / x.shape[-1]), act("ADD_UNARY_SFPU", EPS), act("RSQRT")]
        return ttnn.multiply(x, ss, input_tensor_b_activations=fac, dtype=BF)

    def attention(self, layer, xq, xk, xv, visit):
        t = self.chunk
        qrsg = self.lin(xq, layer.dq.weight)

        def columns(start, stop):
            return ttnn.slice(qrsg, [0, 0, 0, start], [1, 1, t, stop])

        if self._fused_heads:
            q = _heads(qrsg, 0, HQ, threshold=22.627416997969522, tau=self.tau_col if layer.is_global else None)
        else:
            q = self.hclip(ttnn.reshape(columns(0, 3072), [1, t, HQ, HD]), 22.627416997969522)
            if layer.is_global:
                q = ttnn.multiply(f32(q), self.tau, dtype=BF)
            q = ttnn.permute(q, (0, 2, 1, 3))
        r = self.hclip(ttnn.reshape(columns(3072, 3456), [1, t, HQ, DR]), 8.0)
        if layer.is_global:
            r = ttnn.multiply(f32(r), self.tau, dtype=BF)
        sink, gate = columns(3456, 3488), columns(3488, 6560)
        r = ttnn.permute(r, (0, 2, 1, 3))
        r = ttnn.pad(r, [(0, 0), (0, 0), (0, 0), (0, 32 - DR)], 0.0)
        table = self.rel_weights[layer.i] if self._project_distance else None
        distance = r if table is not None else self.c.mm(r, self.rel_weights[layer.i])

        kr, vr = self.lin(xk, layer.wk), self.lin(xv, layer.wv)
        dk = self.lin(xk, layer.k_wproj) if layer.is_global else None
        dv = self.lin(xv, layer.v_wproj) if layer.is_global else None
        k = _stencil(kr, layer.k_taps, visit.conv[0], self.meta, dk)
        v = _stencil(vr, layer.v_taps, visit.conv[1], self.meta, dv)
        if self._fused_heads:
            k = _heads(k, 0, HKV, threshold=22.627416997969522)
            v = _heads(v, 0, HKV, rms=True)
        else:
            k = ttnn.permute(self.hclip(ttnn.reshape(k, [1, t, HKV, HD]), 22.627416997969522), (0, 2, 1, 3))
            v = ttnn.permute(self.hrms(ttnn.reshape(v, [1, t, HKV, HD])), (0, 2, 1, 3))
        out = _tiled_attention(
            q,
            k,
            v,
            visit.k,
            visit.v,
            sink,
            distance,
            self.meta,
            layer.is_global,
            self.attn_out,
            retain_scores=self._retain_scores,
            global_score_tiles=self._global_score_tiles,
            table=table,
        )
        _launch(
            _state_program,
            (_tasks(self.dev, HKV * t // 32 * (HD // 32)), t // 32, visit.klen // 32, HD // 32, False),
            (self.meta, k, visit.k, v, visit.v),
        )
        out = ttnn.multiply(out, gate, input_tensor_b_activations=[act("SIGMOID")], dtype=BF)
        return self.lin(out, layer.wo)

    def router_logits(self, layer, x):
        # Bound the inner dimension accumulated in Tensix DST. Router cutoff
        # margins can be smaller than the accumulation drift of a full-K GEMM.
        pieces = D // _ROUTER_K_BLOCK
        blocks = ttnn.reshape(x, [1, self.chunk, pieces, _ROUTER_K_BLOCK])
        blocks = ttnn.permute(blocks, (0, 2, 1, 3))
        partials = ttnn.matmul(blocks, self.router_weights[layer.i], dtype=FP, compute_kernel_config=self.c.ck_exact)
        return ttnn.sum(partials, dim=1, keepdim=True)

    def moe(self, layer, x):
        logits = self.router_logits(layer, x)
        if layer.expert_stream is not None:
            ws = self.expert_workspace
            experts_stream.route_prefill(logits, layer.router_b, self.meta, ws)
            return experts_stream.moe_prefill(x, None, layer.expert_stream, workspace=ws)

        scores = ttnn.sigmoid(logits)
        _, ids = ttnn.topk(ttnn.add(scores, layer.router_b), k=TOPK, dim=-1)
        gamma = ttnn.gather(scores, -1, ids)
        den = ttnn.add(ttnn.sum(gamma, dim=-1, keepdim=True), 1e-20)
        gamma = ttnn.multiply(ttnn.multiply(gamma, ttnn.reciprocal(den)), 8.0)
        ids = ttnn.to_layout(ttnn.typecast(ids, ttnn.uint32), RM)
        gamma = ttnn.scatter(self.stock_scores, -1, ids, ttnn.to_layout(gamma, RM))
        gamma = ttnn.permute(ttnn.to_layout(gamma, TILE), (0, 3, 2, 1))
        return self.stock_experts(layer, x, gamma)

    def stock_experts(self, layer, x, gamma):
        tiles = self.chunk // 32

        def config(k):
            sb = max(n for n in (1, 2, 4) if tiles % n == 0)
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(9, 8),
                in0_block_w=k,
                out_subblock_h=sb,
                out_subblock_w=1,
                out_block_h=tiles,
                out_block_w=1,
                per_core_M=tiles,
                per_core_N=1,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            )

        common = dict(
            sparsity=self.ops.dummy,
            indices=self.all_ids,
            is_input_b_sparse=True,
            memory_config=DRAM,
            dtype=BF,
            compute_kernel_config=self.c.ck_sparse,
        )
        gu = ttnn.sparse_matmul(x, layer.w_gu, is_input_a_sparse=False, program_config=config(8), **common)
        gu = ttnn.reshape(gu, (1, E, self.chunk, 2 * F))
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, E, self.chunk, F])
        up = ttnn.slice(gu, [0, 0, 0, F], [1, E, self.chunk, 2 * F])
        a = ttnn.multiply(
            gate,
            gate,
            input_tensor_a_activations=[act("SOFTCAP", 4.0)],
            input_tensor_b_activations=[act("SIGMOID")],
            dtype=FP,
        )
        a = ttnn.multiply(a, up, input_tensor_b_activations=[act("SOFTCAP", 25.0)], dtype=BF)
        y = ttnn.sparse_matmul(a, layer.w_dn, is_input_a_sparse=True, program_config=config(6), **common)
        y = ttnn.reshape(y, (1, E, self.chunk, D))
        return bf16(ttnn.sum(ttnn.multiply(y, gamma), dim=1, keepdim=True))

    def gate_msg(self, layer, s, u, m, un=None):
        if not self._message_dense:
            return layer.gate_msg(s, u, m, un)
        dn, wf, wg = layer.gates[s]
        u, m = f32(u), f32(m)
        # Rounding each half before the concat gives the same BF16 operand with less traffic.
        if un is None:
            un16 = bf16(self.ops.rms(u))
        elif self._un16 is not None and self._un16[0] is un:
            un16 = self._un16[1]
        else:
            un16 = bf16(un)
            self._un16 = (un, un16)
        hin = ttnn.concat([un16, bf16(self.ops.rms(m))], dim=-1)
        h = f32(self.lin(hin, dn))
        if self._fused_gate:
            return _message_gate(h, wf, wg, u, m)
        forget = dense.message_prefill_up(h, wf)
        message = dense.message_prefill_up(h, wg)
        kept = ttnn.multiply(u, forget, input_tensor_b_activations=[act("TANH"), act("RSUB", 1.0)])
        return ttnn.add(kept, ttnn.multiply(m, message, input_tensor_b_activations=[act("TANH")]))

    def layer(self, layer, x, bank, visit, messages):
        ops = self.ops
        if layer.is_global:
            qkv = [f32(x)] * 3 if layer.q_attn is None else self.mix(bank, x, layer.q_attn, 3)
            if messages is not None:
                qkv = [self.gate_msg(layer, s, u, v) for s, u, v in zip("qkv", qkv, messages)]
            qkv = [ops.rms(u, w, BF) for u, w in zip(qkv, (layer.norm_q, layer.norm_k, layer.norm_v))]
        else:
            h = f32(x) if layer.q_attn is None else self.mix(bank, x, layer.q_attn, 1)[0]
            if messages is None:
                qkv = [ops.rms(h, layer.norm_in, BF)] * 3
            else:
                hn = ops.rms(h)
                qkv = [ops.rms(self.gate_msg(layer, s, h, v, hn), layer.norm_in, BF) for s, v in zip("qkv", messages)]

        a = self.attention(layer, *qkv, visit)
        a = f32(_stencil(a, layer.pa_taps, visit.conv[2], self.meta))
        if layer.block_boundary:
            bank = [*bank, ops.rms(x)]
            z = a
        else:
            z = ttnn.add(f32(x), a)
        hm = self.mix(bank, z, layer.q_mlp, 1)[0]
        if messages is not None:
            hm = self.gate_msg(layer, "mlp", hm, messages[3])
        y = self.moe(layer, ops.rms(hm, layer.norm_post, BF))
        y = _stencil(y, layer.pm_taps, visit.conv[3], self.meta)
        return bf16(ttnn.add(z, f32(y))), bank

    def _body(self):
        self._bank_items = {}
        self._un16 = None
        try:
            self._body_layers()
        finally:
            self._bank_items = {}
            self._un16 = None

    def _body_layers(self):
        m, layers = self.m, self.m.layers
        x, bank = self.rows, []

        def run_range(start, stop, p, previous=None):
            nonlocal x, bank
            for layer in layers[start:stop]:
                messages = None
                if previous is not None:
                    prev_x, prev_bank = previous
                    messages = self.mix(prev_bank, prev_x, m.msg_q[layer.i - LOOP0], 4)
                x, bank = self.layer(layer, x, bank, m.visits[visit_slot(layer.i, p)], messages)

        run_range(0, LOOP0, 0)
        x_pre, bank_pre = x, bank
        run_range(LOOP0, LOOP1, 0)
        for p in range(1, NPASS):
            previous = x, bank[len(bank_pre) + 1 :]
            x, bank = x_pre, bank_pre
            run_range(LOOP0, LOOP1, p, previous)
        run_range(LOOP1, len(layers), 0)

        h = self.ops.rms(self.mix(bank, x, m.final_q, 1)[0], m.norm, BF)
        _launch(
            _state_program, (_tasks(self.dev, D // 32), self.chunk // 32, 0, D // 32, True), (self.meta, h, self.last)
        )

    def _readout(self):
        ttnn.copy(dense.linear(self.last, self.m.head), self.logits)

    def _stage(self, tokens, pos):
        valid = len(tokens)
        if not 1 <= valid <= self.chunk or pos < 0 or pos % 32 or pos + valid > self.m.lmax:
            raise ValueError("prefill requires a nonempty, in-capacity chunk starting on a tile boundary")
        rows = torch.zeros(self.chunk, D, dtype=torch.bfloat16)
        e = self.m.embed[torch.as_tensor(tokens, dtype=torch.long)].float()
        rows[:valid] = (e * torch.rsqrt(e.square().mean(-1, keepdim=True) + EPS) * self.m.embed_norm).bfloat16()
        tau = 1.0 + LOG_ALPHA * ((pos + 1 + torch.arange(self.chunk)) / LOG_FLOOR).clamp_min(1.0).log()
        meta = torch.tensor([[pos, valid, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
        for host, target in (
            (rows.reshape(1, 1, self.chunk, D), self.rows),
            (meta, self.meta),
            (tau.reshape(1, self.chunk, 1, 1), self.tau),
            (tau.reshape(1, 1, self.chunk, 1).expand(1, 1, self.chunk, 32), self.tau_col),
        ):
            ttnn.copy_host_to_device_tensor(self.c.host(host, target.dtype, target.layout), target)

    def run_chunk(self, tokens, pos, *, return_logits=True):
        """Advance the shared cache and optionally return the last token's logits."""
        self._stage(tokens, pos)
        if self.body_trace is None:
            self._body()
        else:
            ttnn.execute_trace(self.dev, self.body_trace, cq_id=0, blocking=False)
        if not return_logits:
            return None
        if self.head_trace is None:
            self._readout()
        else:
            ttnn.execute_trace(self.dev, self.head_trace, cq_id=0, blocking=False)
        lg = ttnn.to_torch(self.logits).reshape(-1)[:VOCAB].float()
        return 30.0 * torch.tanh(lg / 30.0)

    def prefill(self, tokens):
        """Run a prompt from position zero with this bucket. Returns (last logits, next position)."""
        lg = None
        for pos in range(0, len(tokens), self.chunk):
            lg = self.run_chunk(tokens[pos : pos + self.chunk], pos, return_logits=pos + self.chunk >= len(tokens))
        return lg, len(tokens)


# Measured warm chunk time is about a + b * rows with a / b near 352 rows
# (0.208 s at 128 rows, 1.038 s at 2,048 rows).
_CHUNK_OVERHEAD_ROWS = 352


def prefill_buckets(buckets, tokens):
    """Run a prompt from position zero, choosing for each chunk the bucket with
    the lowest estimated cost for the remaining tokens. Returns (logits, position)."""
    buckets = sorted(buckets, key=lambda p: p.chunk)
    lg, pos, total = None, 0, len(tokens)
    while pos < total:
        rest = total - pos
        pf = min(buckets, key=lambda p: (-(-rest // p.chunk) * (p.chunk + _CHUNK_OVERHEAD_ROWS), -p.chunk))
        n = min(pf.chunk, rest)
        lg = pf.run_chunk(tokens[pos : pos + n], pos, return_logits=pos + n >= total)
        pos += n
    return lg, pos
