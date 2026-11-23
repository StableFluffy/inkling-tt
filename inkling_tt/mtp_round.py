from functools import lru_cache, partial
from pathlib import Path

import torch
import ttnn

from inkling_tt.model import D
from inkling_tt.model2 import RM
from inkling_tt.mtp_block import DraftBlock, DraftHeadProjection
from inkling_tt.mtp_context import DraftContextProjection, DraftPositions
from inkling_tt.mtp_markov import MarkovDraftHead
from inkling_tt.mtp_target import TargetVerify
from inkling_tt.mtp_weights import fold_head, load_drafter
from inkling_tt.prefill import _alloc, _grid, _launch, _program, _tile_cb

_KERNELS = Path(__file__).resolve().parent / "kernels" / "mtp"


@lru_cache(maxsize=8)
def _accept_program(rows, specs):
    grid, _ = _grid(((0, 0, 0, 0),))
    pos, index = specs[0][:2]
    meta, result = specs[0][-2:]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "accept.cpp"),
        core_ranges=grid,
        compile_time_args=[rows, *index, *pos, *meta, *result],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [_tile_cb(grid, 0, ttnn.uint32, size=2 * rows * 64 + 64)])


@lru_cache(maxsize=8)
def _publish_program(specs):
    grid, _ = _grid(((0, 0, 0, 0),))
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "publish.cpp"),
        core_ranges=grid,
        compile_time_args=list(sum(specs[0], ())),
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [_tile_cb(grid, 0, ttnn.uint32, size=128)])


class RoundDecision:
    def __init__(self, device, rows):
        if not 2 <= rows <= 8:
            raise ValueError("A speculative round needs two to eight target rows")
        self.rows = rows
        self.result = _alloc(device, (1, 16), ttnn.uint32, RM)

    def accept(self, position, target, drafts, meta):
        """Write the commit descriptor and the accepted prefix followed by the B1 continuation."""
        if len(target) != self.rows or len(drafts) != self.rows - 1:
            raise ValueError("Target and draft decisions must match the configured round width")
        _launch(_accept_program, (self.rows,), (position, *target, *drafts, meta, self.result))

    def read(self):
        data = ttnn.to_torch(self.result).flatten().tolist()
        return data[2 : 2 + data[0]], data[1]

    def publish(self, anchor, position):
        _launch(_publish_program, (), (self.result, position, anchor))


class SpeculativeRound:
    def __init__(self, model, buckets, snapshot, proposals=3):
        if model.trace_id is not None or not buckets or model.taps is None or not hasattr(model, "embed_table"):
            raise ValueError("Set up speculative decoding with target taps, prefill buckets and device feedback")
        self.model, self.device = model, model.c.dev
        self.rows = proposals + 1
        weights = load_drafter(snapshot)
        fold, bias = fold_head(model.wts.get("lm_head.weight"), weights)
        self.target = TargetVerify(model, proposals, streaming=True)
        self.block = DraftBlock(self.device, weights, self.rows, model.lmax)
        self.projection = DraftHeadProjection(
            self.device,
            model.head,
            weights,
            proposals,
            head_buffers=(self.target.head_input, self.target.head_output),
        )
        self.markov = MarkovDraftHead(self.device, weights, fold, bias, proposals)
        self.positions = DraftPositions(self.device, model.lmax)
        self.decision = RoundDecision(self.device, self.rows)
        self.anchor = _alloc(self.device, (1, 1), ttnn.uint32, RM)
        self.position = _alloc(self.device, (1,), ttnn.int32, RM)
        self.last = _alloc(self.device, (1, 1, 1, D))
        self.prefill_context = {
            bucket: DraftContextProjection(self.block, weights, bucket.chunk, self.last, meta=bucket.meta)
            for bucket in buckets
        }
        self.verify_context = DraftContextProjection(self.block, weights, 32, self.last, meta=self.target.commit.meta)
        self.prefill_traces = {}
        self.trace = None
        self.pending = None
        self.warmed = False

    def can_run(self, position):
        return 0 <= position <= self.model.lmax - self.rows

    def _write(self, token, position):
        for host, dest in (([[token]], self.anchor), ([position], self.position)):
            src = ttnn.from_torch(torch.tensor(host, dtype=torch.int32), dtype=dest.dtype, layout=RM)
            ttnn.copy_host_to_device_tensor(src, dest)

    def _extend(self, context, taps):
        self.positions.rotate(context.meta, context.cos, context.sin)
        context.extend(taps)

    def _forward(self):
        self.positions.prepare(self.block, self.anchor, self.position, self.model.embed_table)
        base, recurrent = self.projection.project(self.block.forward(), self.last)
        drafts, _ = self.markov.score(base, recurrent, self.anchor)
        self.target.stage_device(self.anchor, drafts, self.position)
        _, target_ids = self.target.verify()
        commit = self.target.commit
        self.decision.accept(self.position, target_ids, drafts, commit.meta)
        commit.commit()
        self._extend(self.verify_context, self.target.taps)
        self.decision.publish(self.anchor, self.position)

    def warm(self):
        for bucket, context in self.prefill_context.items():
            self._extend(context, bucket.taps)
        self._write(0, 0)
        self._forward()
        ttnn.synchronize_device(self.device)
        self.model.reset()
        self.warmed = True

    def _capture(self, body):
        trace = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            body()
        except Exception:
            ttnn.end_trace_capture(self.device, trace, cq_id=0)
            ttnn.release_trace(self.device, trace)
            raise
        ttnn.end_trace_capture(self.device, trace, cq_id=0)
        return trace

    def capture(self):
        if not self.warmed or self.trace is not None:
            raise RuntimeError("Speculation needs a warmed graph with no existing round trace")
        try:
            for bucket, context in self.prefill_context.items():
                self.prefill_traces[bucket] = self._capture(partial(self._extend, context, bucket.taps))
            self._write(0, 0)
            self.trace = self._capture(self._forward)
        except Exception:
            self.close()
            raise
        ttnn.synchronize_device(self.device)
        self.model.reset()
        self.pending = None

    def append_prompt(self, bucket):
        self.pending = None
        ttnn.execute_trace(self.device, self.prefill_traces[bucket], cq_id=0, blocking=False)

    def run(self, token, position):
        if self.trace is None or not self.can_run(position):
            raise ValueError("Speculative round requires a captured graph and room for the full block")
        if self.pending != (token, position):
            self._write(token, position)
        self.pending = None
        ttnn.execute_trace(self.device, self.trace, cq_id=0, blocking=False)
        emitted, end = self.decision.read()
        self.pending = (emitted[-1], end)
        self.model.st.device_state = None
        return emitted, end

    def close(self):
        if self.trace is not None:
            ttnn.release_trace(self.device, self.trace)
            self.trace = None
        for trace in self.prefill_traces.values():
            ttnn.release_trace(self.device, trace)
        self.prefill_traces.clear()
        self.pending = None
