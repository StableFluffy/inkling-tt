from functools import lru_cache
from pathlib import Path

import torch
import ttnn

from inkling_tt.model import HD, HKV, D
from inkling_tt.model2 import BF, DRAM, FP, RM, TILE
from inkling_tt.mtp_block import _norm_heads, _norm_rows, _remap_program, _Rope
from inkling_tt.mtp_markov import _embedding_program
from inkling_tt.prefill import _alloc, _grid, _launch, _program, _state_program, _tasks, _tile_cb

_KERNELS = Path(__file__).resolve().parent / "kernels" / "mtp"


@lru_cache(maxsize=32)
def _commit_program(tasks, row_tiles, capacity, specs):
    grid, rt = _grid(tasks)
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "context_state.cpp"),
        core_ranges=grid,
        runtime_args=rt,
        compile_time_args=[row_tiles, capacity, *sum(specs[0], ())],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [_tile_cb(grid, 0, BF)])


@lru_cache(maxsize=16)
def _position_program(tasks, capacity, rows, specs, *, mask=False):
    grid, rt = _grid(tasks)
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "position.cpp"),
        core_ranges=grid,
        runtime_args=rt,
        compile_time_args=[capacity, rows, *sum(specs[0], ())],
        defines=[("DRAFT_MASK", "1")] if mask else [],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [_tile_cb(grid, 0, FP)])


class DraftPositions:
    def __init__(self, device, lmax):
        self.device, self.lmax = device, lmax
        inv_freq = 1 / (1_000_000 ** (torch.arange(0, HD, 2).float() / HD))
        angles = torch.arange(lmax)[:, None] * inv_freq
        table = torch.cat((angles.cos(), angles.sin()), dim=-1)
        self.table = ttnn.from_torch(table, dtype=FP, layout=RM, device=device, memory_config=DRAM)

    def rotate(self, position, cos, sin):
        rows = cos.shape[-2]
        _launch(
            _position_program,
            (_tasks(self.device, rows // 8), self.lmax, rows),
            (position, self.table, cos, sin),
        )

    def prepare(self, block, anchor, position, embedding):
        _launch(_embedding_program, (D, 0), (anchor, embedding, block.embedded))
        self.rotate(position, block.cos, block.sin)
        for mask in block.mask:
            tiles = mask.shape[-1] // 32
            _launch(
                _position_program,
                (_tasks(self.device, tiles), mask.shape[-1] - 32, block.rows),
                (position, mask),
                mask=True,
            )


class DraftContextProjection:
    """Project committed target taps and write only the valid rows to draft KV."""

    def __init__(self, block, weights, chunk, last, *, meta=None):
        if chunk < 32 or chunk % 32:
            raise ValueError("DSpark context projection needs tile-aligned chunks")
        device = block.hidden.device()
        self.device, self.chunk, self.block = device, chunk, block
        self.last = last
        self.alpha = weights["fuse_weight"].float().softmax(-1).bfloat16().tolist()

        def upload(x):
            return ttnn.from_torch(x.contiguous(), dtype=BF, layout=TILE, device=device, memory_config=DRAM)

        self.norm = []
        self.wk, self.wv, self.kn = [], [], []
        for layer in range(3):
            prefix = f"layers.{layer}."
            attn = prefix + "self_attn."
            self.norm.append(upload(weights[prefix + "dflare_norm.weight"].reshape(1, 1, 1, D)))
            self.wk.append(upload(weights[attn + "k_proj_ctx.weight"].T))
            self.wv.append(upload(weights[attn + "v_proj_ctx.weight"].T))
            self.kn.append(upload(weights[attn + "k_norm_ctx.weight"].reshape(1, 1, 1, HD)))
        self.norm.append(upload(weights["markov_head.dflare_norm.weight"].reshape(1, 1, 1, D)))

        row = lambda width, dtype=BF: _alloc(device, (1, 1, chunk, width), dtype)
        self.meta = meta if meta is not None else _alloc(device, (1, 8), ttnn.uint32, RM)
        self.cos, self.sin = (row(HD // 2, FP) for _ in range(2))
        self.acc, self.temp = (row(D, FP) for _ in range(2))
        self.normalized = row(D)
        self.k, self.v, self.k_norm = (row(HKV * HD) for _ in range(3))
        self.k_heads, self.v_heads, self.k_rot = (_alloc(device, (1, HKV, chunk, HD)) for _ in range(3))
        self.rope = _Rope(device, HKV, chunk)

    def _mix(self, features, i):
        ttnn.mul(features[0], self.alpha[i][0], dtype=FP, output_tensor=self.acc)
        for source, alpha in zip(features[1:], self.alpha[i][1:]):
            ttnn.mul(source, alpha, dtype=FP, output_tensor=self.temp)
            ttnn.add_(self.acc, self.temp)
        _norm_rows(self.acc, self.norm[i], self.normalized, self.chunk)

    def extend(self, features):
        row_tiles = self.chunk // 32
        cols = HD // 32
        tasks = _tasks(self.device, HKV * row_tiles * cols)
        for i, (cache_k, cache_v) in enumerate(self.block.cache):
            self._mix(features, i)
            for weight, dest in ((self.wk[i], self.k), (self.wv[i], self.v)):
                ttnn.linear(self.normalized, weight, compute_kernel_config=self.block.ck, optional_output_tensor=dest)
            _norm_heads(self.k, self.kn[i], self.k_norm, self.chunk, HKV)
            for source, dest in ((self.k_norm, self.k_heads), (self.v, self.v_heads)):
                _launch(_remap_program, (tasks, cols, cols, row_tiles, HKV), (source, dest), heads=True)
            self.rope.apply(self.k_heads, self.cos, self.sin, self.k_rot)
            for source, cache in ((self.k_rot, cache_k), (self.v_heads, cache_v)):
                _launch(_commit_program, (tasks, row_tiles, cache.shape[-2] - 32), (self.meta, source, cache))
        self._mix(features, 3)
        _launch(
            _state_program,
            (_tasks(self.device, D // 32), row_tiles, 0, D // 32, True),
            (self.meta, self.normalized, self.last),
        )
