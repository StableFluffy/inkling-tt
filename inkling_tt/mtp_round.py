from functools import lru_cache, partial
from pathlib import Path
from statistics import median
from time import perf_counter

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
        """Write the accepted range and its IDs, including the B1 continuation."""
        rows = len(target)
        if not 2 <= rows <= self.rows or len(drafts) != rows - 1:
            raise ValueError("Target and draft decisions must match the active round width")
        _launch(_accept_program, (rows,), (position, *target, *drafts, meta, self.result))

    def read(self, result=None):
        data = ttnn.to_torch(self.result if result is None else result).flatten().tolist()
        return data[2 : 2 + data[0]], data[1]

    def publish(self, anchor, position):
        _launch(_publish_program, (), (self.result, position, anchor))


class SpeculativeRound:
    def __init__(self, model, buckets, snapshot, proposals=3, *, adaptive=False):
        if model.trace_id is not None or not buckets or model.taps is None or not hasattr(model, "embed_table"):
            raise ValueError("Set up speculative decoding with target taps, prefill buckets and device feedback")
        self.model, self.device = model, model.c.dev
        self.rows = proposals + 1
        self.adaptive = adaptive
        self.widths = tuple(range(2, min(self.rows, 4) + 1)) if adaptive else (self.rows,)
        self.costs = {}
        self.start_request()
        weights = load_drafter(snapshot)
        fold, bias = fold_head(model.wts.get("lm_head.weight"), weights)
        self.target = TargetVerify(model, proposals, streaming=True)
        self.block = DraftBlock(self.device, weights, self.rows, model.lmax, short_prefix=True)
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
        self.verify_context = DraftContextProjection(self.block, weights, 32, self.last)
        self.prefill_traces = {}
        self.draft_traces = {}
        self.verify_traces = {}
        self.context_trace = None
        self.pending = None
        self.warmed = False

    def start_request(self):
        self.accepted, self.rejected = 1.7, 0.3
        self.plain = False
        self.pending = None

    def can_run(self, position, rows=None):
        rows = self.rows if rows is None else rows
        return 0 <= position <= self.model.lmax - rows

    def _short(self, position):
        return self.block.short_attention is not None and position <= 1024

    def pick(self, position, limit):
        if self.plain:
            return 1
        short = self._short(position)
        acceptance = self.accepted / (self.accepted + self.rejected)
        widths = (r for r in self.widths if r <= limit and self.can_run(position, r))
        if self.adaptive:
            widths = (1, *widths)
            width = max(widths, key=lambda r: sum(acceptance**i for i in range(r)) / self.costs[short, r])
        else:
            width = next(widths, 1)
        self.plain = width == 1
        return width

    def _write(self, token, position):
        for host, dest in (([[token]], self.anchor), ([position], self.position)):
            src = ttnn.from_torch(torch.tensor(host, dtype=torch.int32), dtype=dest.dtype, layout=RM)
            ttnn.copy_host_to_device_tensor(src, dest)

    def _extend(self, context, taps):
        self.positions.rotate(context.meta, context.cos, context.sin)
        context.extend(taps)

    def _draft(self, short, rows):
        self.positions.prepare(self.block, self.anchor, self.position, self.model.embed_table, short=short)
        base, recurrent = self.projection.project(self.block.forward(short=short), self.last)
        self.markov.score(base[: rows - 1], recurrent[: rows - 1], self.anchor)

    def _verify(self, rows):
        drafts = self.markov.ids[: rows - 1]
        self.target.stage_device(self.anchor, drafts, self.position)
        _, target_ids = self.target.verify(rows=rows)
        self.decision.accept(self.position, target_ids, drafts, self.verify_context.meta)
        self.decision.publish(self.anchor, self.position)

    def _context(self):
        self._extend(self.verify_context, self.target.taps)

    def warm(self):
        for bucket, context in self.prefill_context.items():
            self._extend(context, bucket.taps)
        for rows in self.widths:
            self._write(0, 0)
            for short in self._draft_widths():
                self._draft(short, rows)
            self._verify(rows)
            self._context()
        ttnn.synchronize_device(self.device)
        self.model.reset()
        self.warmed = True

    def _draft_widths(self):
        return (False, True) if self.block.short_attention is not None else (False,)

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
        if not self.warmed or self.verify_traces:
            raise RuntimeError("Speculation needs a warmed graph with no existing round trace")
        try:
            for bucket, context in self.prefill_context.items():
                self.prefill_traces[bucket] = self._capture(partial(self._extend, context, bucket.taps))
            for rows in self.widths:
                self._write(0, 0)
                for short in self._draft_widths():
                    self.draft_traces[short, rows] = self._capture(partial(self._draft, short, rows))
                self.verify_traces[rows] = self._capture(partial(self._verify, rows))
            self.context_trace = self._capture(self._context)
            if self.adaptive:
                self._measure_costs()
        except Exception:
            self.close()
            raise
        ttnn.synchronize_device(self.device)
        self.model.reset()
        self.pending = None

    def _measure_costs(self):
        for short in self._draft_widths():
            position = min(
                29 if short or self.block.short_attention is None else self.model.lmax - 32,
                self.model.lmax - max(self.widths),
            )
            variants = (1, *self.widths)
            samples = {rows: [] for rows in variants}
            for trial in range(4):
                for i in range(len(variants)):
                    rows = variants[(i + trial) % len(variants)]
                    if rows == 1:
                        self.model.st.write(self.model.embed_row(0), position)
                        traces = (self.model.trace_id,)
                    else:
                        self._write(0, position)
                        traces = (self.draft_traces[short, rows], self.verify_traces[rows], self.context_trace)
                    ttnn.synchronize_device(self.device)
                    start = perf_counter()
                    for trace in traces:
                        ttnn.execute_trace(self.device, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(self.device)
                    samples[rows].append(perf_counter() - start)
            for rows in variants:
                self.costs[short, rows] = median(samples[rows][1:])

    def append_prompt(self, bucket):
        self.pending = None
        ttnn.execute_trace(self.device, self.prefill_traces[bucket], cq_id=0, blocking=False)

    def run(self, token, position, rows=None):
        rows = self.rows if rows is None else rows
        if rows not in self.verify_traces or not self.can_run(position, rows):
            raise ValueError("Speculative round requires a captured graph and room for the full block")
        if self.pending != (token, position):
            self._write(token, position)
        self.pending = None
        short = self._short(position)
        ttnn.execute_trace(self.device, self.draft_traces[short, rows], cq_id=0, blocking=False)
        ttnn.execute_trace(self.device, self.verify_traces[rows], cq_id=0, blocking=False)
        result = ttnn.from_device(self.decision.result, blocking=False)
        ready = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.context_trace, cq_id=0, blocking=False)
        ttnn.event_synchronize(ready)
        emitted, end = self.decision.read(result)
        accepted = len(emitted) - 1
        self.accepted = 0.9 * self.accepted + accepted
        self.rejected = 0.9 * self.rejected + int(accepted < rows - 1)
        self.pending = (emitted[-1], end)
        self.model.st.device_state = None
        return emitted, end

    def close(self):
        if self.context_trace is not None:
            ttnn.release_trace(self.device, self.context_trace)
            self.context_trace = None
        for trace in self.verify_traces.values():
            ttnn.release_trace(self.device, trace)
        self.verify_traces.clear()
        for trace in self.draft_traces.values():
            ttnn.release_trace(self.device, trace)
        self.draft_traces.clear()
        for trace in self.prefill_traces.values():
            ttnn.release_trace(self.device, trace)
        self.prefill_traces.clear()
        self.pending = None
