import math
from functools import lru_cache
from pathlib import Path

import torch
import ttnn

from inkling_tt.model import VOCAB
from inkling_tt.model2 import BF, DRAM, FP, RM, TILE, act
from inkling_tt.mtp_weights import PADDED_VOCAB, RANK
from inkling_tt.prefill import _alloc, _compute, _copy_into, _grid, _launch, _program, _tasks, _tile_cb

_KERNELS = Path(__file__).resolve().parent / "kernels" / "mtp"


@lru_cache(maxsize=8)
def _logit_row_program(tasks, fp32, specs):
    grid, rt = _grid(tasks)
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "logit_row.cpp"),
        core_ranges=grid,
        runtime_args=rt,
        compile_time_args=[VOCAB, 2 * PADDED_VOCAB, int(fp32), *sum(specs[0], ())],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [_tile_cb(grid, 0, BF, size=192 if fp32 else 128)])


@lru_cache(maxsize=4)
def _embedding_program(width, offset, specs):
    grid, _ = _grid(((0, 0, 0, 0),))
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "markov_embedding.cpp"),
        core_ranges=grid,
        compile_time_args=[width, offset, *sum(specs[0], ())],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [_tile_cb(grid, 0, BF, size=2 * width)])


def _slice_columns(src, start, dest):
    ttnn.slice(src, [0, 0, 0, start], [1, 1, 1, start + dest.shape[-1]], output_tensor=dest)


@lru_cache(maxsize=16)
def _cast_program(tasks, specs):
    grid, rt = _grid(tasks)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "cast_io.cpp"),
            core_ranges=grid,
            runtime_args=rt,
            compile_time_args=list(specs[0][0]),
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "cast_compute.cpp"),
            core_ranges=grid,
            runtime_args=rt,
            config=_compute((0,)),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "cast_io.cpp"),
            core_ranges=grid,
            runtime_args=rt,
            compile_time_args=list(specs[2][0]),
            defines=[("DRAFT_WRITE", "1")],
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    return _program(kernels, [_tile_cb(grid, 0, FP, 2), _tile_cb(grid, 16, BF, 2)])


def _cast(src, dest):
    if src.dtype == FP and dest.dtype == BF:
        tiles = math.prod(src.padded_shape) // 1024
        _launch(_cast_program, (_tasks(src.device(), tiles),), (src,), (), (dest,))
    else:
        ttnn.assign(src, dtype=dest.dtype, memory_config=dest.memory_config(), output_tensor=dest)


class MarkovDraftHead:
    """Score a fixed number of proposals from BF16 base logits and FP32 recurrent context rows."""

    def __init__(self, device, weights, fold, fold_bias, proposals):
        if not 1 <= proposals <= 7:
            raise ValueError("Markov draft width must be between one and seven")
        self.device = device
        self.proposals = proposals
        self.fused_scores = True
        self.compute = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )

        def upload(data, layout=TILE):
            return ttnn.from_torch(data.contiguous(), dtype=BF, layout=layout, device=device, memory_config=DRAM)

        markov = "markov_head."
        self.prev_weight = upload(weights[markov + "markov_w1.weight"], RM)
        self.state_weight = upload(weights[markov + "rnn_proj.weight"][:, : 2 * RANK].T)
        self.fold = upload(torch.nn.functional.pad(fold.T, (0, PADDED_VOCAB - VOCAB)))
        self.fold_bias = upload(torch.nn.functional.pad(fold_bias, (0, PADDED_VOCAB - VOCAB)).reshape(1, 1, 1, -1))
        self.zero = upload(torch.zeros(1, 1, 1, RANK))
        row = lambda width, dtype=BF: _alloc(device, (1, 1, 1, width), dtype)
        self.input = row(2 * RANK)
        self.state = row(RANK)
        self.raw = row(3 * RANK, FP)
        self.raw_bf = row(3 * RANK)
        self.gate, self.candidate, self.latent = (row(RANK) for _ in range(3))
        self.keep, self.replace, self.retained, self.new = (row(RANK) for _ in range(4))
        self.nonlinear_arg, self.nonlinear_out = (row(RANK, FP) for _ in range(2))
        self.correction = row(PADDED_VOCAB, FP)
        self.biased = row(PADDED_VOCAB)
        self.summed = row(PADDED_VOCAB)
        self.scaled, self.capped = (row(PADDED_VOCAB, FP) for _ in range(2))
        self.logits = tuple(_alloc(device, (1, 1, 1, PADDED_VOCAB), BF, RM) for _ in range(proposals))
        self.ids = tuple(_alloc(device, (1, 1, 1, 1), ttnn.uint32, RM) for _ in range(proposals))

    def _linear(self, x, weight, dest):
        ttnn.linear(x, weight, dtype=dest.dtype, compute_kernel_config=self.compute, optional_output_tensor=dest)

    def _nonlinear(self, op, t, out):
        _cast(t, self.nonlinear_arg)
        op(self.nonlinear_arg, output_tensor=self.nonlinear_out)
        _cast(self.nonlinear_out, out)

    def _recur(self, prev, context):
        _copy_into(self.state, self.input, 0)
        _launch(_embedding_program, (RANK, RANK // 32), (prev, self.prev_weight, self.input))
        self._linear(self.input, self.state_weight, self.raw)
        ttnn.add(self.raw, context, dtype=BF, output_tensor=self.raw_bf)
        _slice_columns(self.raw_bf, 0, self.gate)
        _slice_columns(self.raw_bf, RANK, self.candidate)
        _slice_columns(self.raw_bf, 2 * RANK, self.latent)
        self._nonlinear(ttnn.sigmoid, self.gate, self.keep)
        self._nonlinear(ttnn.tanh, self.candidate, self.candidate)
        self._nonlinear(ttnn.tanh, self.latent, self.latent)
        ttnn.rsub(self.keep, 1.0, output_tensor=self.replace)
        ttnn.mul(self.keep, self.state, output_tensor=self.retained)
        ttnn.mul(self.replace, self.candidate, output_tensor=self.new)
        ttnn.add(self.retained, self.new, output_tensor=self.state)

    def score(self, base_rows, context_rows, anchor):
        """Enqueue the recurrent scoring steps. The anchor is a device UINT32 [1,1] tensor."""
        if len(base_rows) != self.proposals or len(context_rows) != self.proposals:
            raise ValueError("Markov inputs must match the configured proposal count")
        ttnn.copy(self.zero, self.state)
        prev = anchor
        for i, (base, context) in enumerate(zip(base_rows, context_rows)):
            self._recur(prev, context)
            self._linear(self.latent, self.fold, self.correction)
            ttnn.add(self.correction, self.fold_bias, dtype=BF, output_tensor=self.biased)
            ttnn.add(base, self.biased, output_tensor=self.summed)
            if self.fused_scores:
                ttnn.mul(
                    self.summed,
                    1.0,
                    dtype=FP,
                    input_tensor_a_activations=[act("SOFTCAP", 30.0)],
                    output_tensor=self.capped,
                )
                logits = self.capped
            else:
                ttnn.mul(self.summed, 1 / 30, dtype=FP, output_tensor=self.scaled)
                ttnn.tanh(self.scaled, output_tensor=self.capped)
                ttnn.mul(self.capped, 30.0, dtype=BF, output_tensor=self.summed)
                logits = self.summed
            _launch(
                _logit_row_program,
                (_tasks(self.device, PADDED_VOCAB // 32), self.fused_scores),
                (logits, self.logits[i]),
            )
            ttnn.argmax(self.logits[i], dim=-1, keepdim=True, output_tensor=self.ids[i])
            prev = self.ids[i]
        return self.ids, self.logits
