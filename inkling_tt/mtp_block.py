import math
from functools import lru_cache
from pathlib import Path

import torch
import ttnn

from inkling_tt import dense, rowvm
from inkling_tt.model import HD, HKV, HQ, D
from inkling_tt.model2 import BF, DRAM, FP, TILE
from inkling_tt.mtp_markov import _cast
from inkling_tt.mtp_projection import _tile_row
from inkling_tt.mtp_weights import MLP_WIDTH, PADDED_VOCAB, RANK
from inkling_tt.prefill import _alloc, _grid, _launch, _program, _tasks, _tile_cb

_KERNELS = Path(__file__).resolve().parent / "kernels" / "mtp"


@lru_cache(maxsize=16)
def _remap_program(tasks, tiles, src_stride, dst_stride, dst_start, specs, *, heads=False):
    grid, rt = _grid(tasks)
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "remap_tiles.cpp"),
        core_ranges=grid,
        runtime_args=rt,
        compile_time_args=[tiles, src_stride, dst_stride, dst_start, *sum(specs[0], ())],
        defines=[("DRAFT_HEADS", "1")] if heads else [],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [_tile_cb(grid, 0, BF)])


def _remap(src, dst, groups, tiles, dst_stride, dst_start):
    args = (_tasks(src.device(), groups * tiles), tiles, tiles, dst_stride, dst_start)
    _launch(_remap_program, args, (src, dst))


def _tile_view(x, shape):
    return ttnn.experimental.alias_view(x, ttnn.Shape(shape), x.dtype)


@lru_cache(maxsize=8)
def _head_copy_program(tasks, shard_tiles, total_tiles, grid_width, specs, *, rows=0):
    grid, _ = _grid(tasks)
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "base_head.cpp"),
        core_ranges=grid,
        compile_time_args=[shard_tiles, total_tiles, grid_width, *specs[0][1]],
        defines=[("DRAFT_BASE_ROWS", str(rows))] if rows else [],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [])


def _head_copy_args(device, handle):
    shard_tiles = handle.output_shards.shard_spec.shape[1] // 32
    total_tiles = handle.n // 32
    workers = (total_tiles + shard_tiles - 1) // shard_tiles
    return _tasks(device, workers), shard_tiles, total_tiles, device.compute_with_storage_grid_size().x


def _norm_rows(x, weight, out, rows):
    chunk, stripes = rowvm._stripes(D)
    grid = x.device().compute_with_storage_grid_size()
    step = grid.x * grid.y // stripes
    for start in range(0, rows, step):
        rowvm._run(
            (rowvm._RMS_STRIPED, chunk, 3, 0, D, stripes),
            [rowvm._stripe(x, chunk, D, row=start, row_step=1), rowvm._stripe(weight, chunk, D)],
            [rowvm._stripe(out, chunk, D, row=start, row_step=1)],
            min(rows - start, step) * stripes,
            task_cols=stripes,
        )


def _norm_heads(x, weight, out, rows, heads):
    grid = x.device().compute_with_storage_grid_size()
    step = grid.x * grid.y // heads
    for start in range(0, rows, step):
        rowvm._run(
            (rowvm._RMS, HD, 3),
            [rowvm._Region(x, HD, row=start, row_step=1, col_step=HD), rowvm._Region(weight, HD)],
            [rowvm._Region(out, HD, row=start, row_step=1, col_step=HD)],
            min(rows - start, step) * heads,
            task_cols=heads,
        )


class _Rope:
    def __init__(self, device, heads, rows=32):
        self.heads, self.rows = heads, rows
        shape = (1, heads, rows, HD // 2)
        self.a, self.b, self.left, self.right = (_alloc(device, shape) for _ in range(4))
        self.x, self.y, self.u, self.v, self.result = (_alloc(device, shape, FP) for _ in range(5))

    def apply(self, src, cos, sin, out):
        start = [0, 0, 0, 0]
        middle = [1, self.heads, self.rows, HD // 2]
        ttnn.slice(src, start, middle, output_tensor=self.a)
        ttnn.slice(src, [0, 0, 0, HD // 2], [1, self.heads, self.rows, HD], output_tensor=self.b)
        _cast(self.a, self.x)
        _cast(self.b, self.y)
        ttnn.mul(self.x, cos, output_tensor=self.u)
        ttnn.mul(self.y, sin, output_tensor=self.v)
        ttnn.sub(self.u, self.v, output_tensor=self.result)
        _cast(self.result, self.left)
        ttnn.mul(self.y, cos, output_tensor=self.u)
        ttnn.mul(self.x, sin, output_tensor=self.v)
        ttnn.add(self.u, self.v, output_tensor=self.result)
        _cast(self.result, self.right)
        groups = self.heads * (self.rows // 32)
        _remap(self.left, out, groups, HD // 64, HD // 32, 0)
        _remap(self.right, out, groups, HD // 64, HD // 32, HD // 64)


class DraftBlock:
    """Device transformer on one DSpark block with fixed-size prefix storage."""

    def __init__(self, device, weights, rows, prefix_capacity, *, sharded_mlp=True):
        if not 2 <= rows <= 8 or prefix_capacity < 32 or prefix_capacity % 32:
            raise ValueError("DSpark needs 2-8 rows and a positive, tile-aligned prefix capacity")
        self.rows = rows
        self.ck = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )

        def upload(x):
            return ttnn.from_torch(x.contiguous(), dtype=BF, layout=TILE, device=device, memory_config=DRAM)

        self.weights = {}
        self.mlp_work = {}
        for name, weight in weights.items():
            if (name.startswith("layers.") and "ctx" not in name and "dflare" not in name) or name == "norm.weight":
                if sharded_mlp and ".mlp." in name:
                    handle = dense.prepare_weight(device, weight.T)
                    self.weights[name] = handle
                    shape = (handle.k, handle.n)
                    if shape not in self.mlp_work:
                        args = _head_copy_args(device, handle)
                        bufs = tuple(
                            ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 32, width)), BF, TILE, device, mem)
                            for width, mem in ((handle.k, handle.input_shards), (handle.n, handle.output_shards))
                        )
                        self.mlp_work[shape] = (*bufs, args)
                else:
                    self.weights[name] = upload(weight.T if weight.ndim == 2 else weight.reshape(1, 1, 1, -1))
        row = lambda n, dtype=BF: _alloc(device, (1, 1, 32, n), dtype)
        self.embedded = upload(weights["mask_embed"].reshape(1, 1, 1, D).expand(1, 1, 32, D))
        self.residual, self.branch, self.hidden = (row(D) for _ in range(3))
        self.norm = upload(torch.zeros(1, 1, 32, D))
        self.merged = row(D, FP)
        self.cos, self.sin = (row(HD // 2, FP) for _ in range(2))
        self.q, self.k, self.v = (row(heads * HD) for heads in (HQ, HKV, HKV))
        self.q_norm, self.k_norm = (upload(torch.zeros(1, 1, 32, heads * HD)) for heads in (HQ, HKV))
        self.q_head = _tile_view(self.q_norm, (1, HQ, 32, HD))
        self.k_head = _tile_view(self.k_norm, (1, HKV, 32, HD))
        self.v_head = _tile_view(self.v, (1, HKV, 32, HD))
        self.q_rot = _alloc(device, (1, HQ, 32, HD))
        self.k_rot = _alloc(device, (1, HKV, 32, HD))
        self.q_group = _tile_view(self.q_rot, (1, HKV, HQ // HKV * 32, HD))
        self.attn = _alloc(device, self.q_group.shape)
        self.attn_flat = _tile_view(self.attn, (1, 1, 32, HQ * HD))
        self.rope_q = _Rope(device, HQ)
        self.rope_k = _Rope(device, HKV)
        self.gate, self.up, self.silu = (row(MLP_WIDTH) for _ in range(3))
        self.gate_fp, self.sigmoid, self.silu_fp = (row(MLP_WIDTH, FP) for _ in range(3))

        self.cache, self.mask, self.scores, self.probs = [], [], {}, {}
        for cap in (min(prefix_capacity, 1024), prefix_capacity, min(prefix_capacity, 1024)):
            width = cap + 32
            self.cache.append(tuple(upload(torch.zeros(1, HKV, width, HD)) for _ in range(2)))
            self.mask.append(_alloc(device, (1, 1, 1, width), FP))
            if cap not in self.scores:
                self.scores[cap] = tuple(_alloc(device, (1, HKV, HQ // HKV * 32, width), FP) for _ in range(2))
                self.probs[cap] = _alloc(device, (1, HKV, HQ // HKV * 32, width))

    def _linear(self, x, name, dest):
        weight = self.weights[name + ".weight"]
        if not isinstance(weight, dense.DenseWeight):
            ttnn.linear(x, weight, compute_kernel_config=self.ck, optional_output_tensor=dest)
            return
        inputs, output, copy_args = self.mlp_work[(weight.k, weight.n)]
        ttnn.interleaved_to_sharded(x, weight.input_shards, preallocated_output=inputs)
        ttnn.linear(
            inputs,
            weight.tensor,
            program_config=weight.program,
            compute_kernel_config=weight.compute,
            memory_config=weight.output_shards,
            optional_output_tensor=output,
        )
        _launch(_head_copy_program, copy_args, (output, dest))

    def _residual(self):
        ttnn.add(self.residual, self.branch, dtype=FP, output_tensor=self.merged)
        _cast(self.merged, self.residual)

    def _attention(self, layer, name):
        for name_part, out in (("q_proj", self.q), ("k_proj", self.k), ("v_proj", self.v)):
            self._linear(self.norm, name + name_part, out)
        _norm_heads(self.q, self.weights[name + "q_norm.weight"], self.q_norm, self.rows, HQ)
        _norm_heads(self.k, self.weights[name + "k_norm.weight"], self.k_norm, self.rows, HKV)
        self.rope_q.apply(self.q_head, self.cos, self.sin, self.q_rot)
        self.rope_k.apply(self.k_head, self.cos, self.sin, self.k_rot)
        keys, values = self.cache[layer]
        cap = keys.shape[-2] - 32
        stride = (cap + 32) * HD // 1024
        for block, cache in ((self.k_rot, keys), (self.v_head, values)):
            _remap(block, cache, HKV, HD // 32, stride, stride - HD // 32)
        scores, masked = self.scores[cap]
        ttnn.matmul(
            self.q_group, keys, transpose_b=True, dtype=FP, compute_kernel_config=self.ck, optional_output_tensor=scores
        )
        ttnn.mul(scores, 1 / math.sqrt(HD), output_tensor=masked)
        ttnn.add_(masked, self.mask[layer])
        ttnn.softmax_in_place(masked, compute_kernel_config=self.ck)
        _cast(masked, self.probs[cap])
        ttnn.matmul(self.probs[cap], values, compute_kernel_config=self.ck, optional_output_tensor=self.attn)
        self._linear(self.attn_flat, name + "o_proj", self.branch)

    def forward(self):
        ttnn.copy(self.embedded, self.residual)
        for layer in range(3):
            prefix = f"layers.{layer}."
            source = self.residual if layer == 0 else self.merged
            _norm_rows(source, self.weights[prefix + "input_layernorm.weight"], self.norm, self.rows)
            self._attention(layer, prefix + "self_attn.")
            self._residual()
            _norm_rows(self.merged, self.weights[prefix + "post_attention_layernorm.weight"], self.norm, self.rows)
            self._linear(self.norm, prefix + "mlp.gate_proj", self.gate)
            self._linear(self.norm, prefix + "mlp.up_proj", self.up)
            _cast(self.gate, self.gate_fp)
            ttnn.sigmoid(self.gate_fp, output_tensor=self.sigmoid)
            ttnn.mul(self.gate_fp, self.sigmoid, output_tensor=self.silu_fp)
            ttnn.mul(self.silu_fp, self.up, dtype=BF, output_tensor=self.silu)
            self._linear(self.silu, prefix + "mlp.down_proj", self.branch)
            self._residual()
        _norm_rows(self.merged, self.weights["norm.weight"], self.hidden, self.rows)
        return self.hidden


class DraftHeadProjection:
    def __init__(self, device, head, weights, proposals, *, direct=True, head_buffers=None):
        self.head = head
        self.idx = range(1, proposals + 1)
        if head_buffers is None:
            head_buffers = tuple(
                ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 32, width)), BF, TILE, device, mem)
                for width, mem in ((D, head.input_shards), (PADDED_VOCAB, head.output_shards))
            )
        self.hidden_shards, self.head_shards = head_buffers
        self.base = None if direct else _alloc(device, self.head_shards.shape)
        self.copy_args = _head_copy_args(device, head)
        width = 3 * RANK
        self.hidden_fixed, self.fixed = (_alloc(device, (1, 1, 32, width), FP) for _ in range(2))
        self.last_fixed = _alloc(device, (1, 1, 1, width), FP)
        self.context_bias = _alloc(device, self.last_fixed.shape, FP)
        self.base_rows = tuple(_alloc(device, (1, 1, 1, PADDED_VOCAB)) for _ in self.idx)
        self.context_rows = tuple(_alloc(device, self.last_fixed.shape, FP) for _ in self.idx)
        rnn = weights["markov_head.rnn_proj.weight"]
        self.hidden_weight, self.last_weight = (
            ttnn.from_torch(data.T.contiguous(), dtype=BF, layout=TILE, device=device, memory_config=DRAM)
            for data in (rnn[:, 2 * RANK : 2 * RANK + D], rnn[:, -D:])
        )
        bias = weights["markov_head.rnn_proj.bias"].float().reshape(1, 1, 1, -1)
        self.bias = ttnn.from_torch(bias, dtype=FP, layout=TILE, device=device, memory_config=DRAM)

    def project(self, hidden, last):
        head = self.head
        ttnn.interleaved_to_sharded(hidden, head.input_shards, preallocated_output=self.hidden_shards)
        ttnn.linear(
            self.hidden_shards,
            head.tensor,
            program_config=head.program,
            compute_kernel_config=head.compute,
            memory_config=head.output_shards,
            optional_output_tensor=self.head_shards,
        )
        if self.base is None:
            _launch(_head_copy_program, self.copy_args, (self.head_shards, *self.base_rows), rows=len(self.base_rows))
        else:
            _launch(_head_copy_program, self.copy_args, (self.head_shards, self.base))
        for rows, w, dst in (
            (hidden, self.hidden_weight, self.hidden_fixed),
            (last, self.last_weight, self.last_fixed),
        ):
            ttnn.linear(rows, w, dtype=FP, compute_kernel_config=head.compute, optional_output_tensor=dst)
        ttnn.add(self.last_fixed, self.bias, output_tensor=self.context_bias)
        ttnn.add(self.hidden_fixed, self.context_bias, output_tensor=self.fixed)
        for i, base, context in zip(self.idx, self.base_rows, self.context_rows):
            if self.base is not None:
                ttnn.slice(self.base, [0, 0, i, 0], [1, 1, i + 1, PADDED_VOCAB], output_tensor=base)
            _tile_row(self.fixed, context, source=i)
        return self.base_rows, self.context_rows
