import argparse
import statistics
import time
from collections import Counter

import torch
import ttnn

from inkling_tt import paths
from inkling_tt.model import D
from inkling_tt.model2 import RM, InklingTT2
from inkling_tt.mtp_block import DraftBlock, DraftHeadProjection, _tile_view
from inkling_tt.mtp_context import DraftContextProjection, DraftPositions
from inkling_tt.mtp_markov import MarkovDraftHead
from inkling_tt.mtp_weights import fold_head, load_drafter
from inkling_tt.prefill import Prefill, _alloc
from serve.runtime import DEVICE_PARAMS, default_precision


def main(steps):
    proposals = 3
    lmax = 2048
    data = torch.load(paths.golden("eval3x128.pt"), weights_only=False)
    device = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    try:
        model = InklingTT2(device, paths.hf_sd(), default_precision(), lmax=lmax, capture_taps=True)
        prefill = Prefill(model, 128)
        weights = load_drafter(paths.snapshot())
        fold, bias = fold_head(model.wts.get("lm_head.weight"), weights)
        block = DraftBlock(device, weights, proposals + 1, lmax)
        projection = DraftHeadProjection(device, model.head, weights, proposals)
        markov = MarkovDraftHead(device, weights, fold, bias, proposals)
        positions = DraftPositions(device, lmax)
        last = _alloc(device, (1, 1, 1, D))
        anchor = _alloc(device, (1, 1), ttnn.uint32, RM)
        position = _alloc(device, (1,), ttnn.int32, RM)
        prompt_context = DraftContextProjection(block, weights, prefill.chunk, last)
        decode_context = DraftContextProjection(block, weights, 32, last)
        decode_taps = tuple(_tile_view(t, (1, 1, 32, D)) for t in model.taps)

        def stage(data, dest):
            host = ttnn.from_torch(torch.tensor(data, dtype=torch.int32), dtype=dest.dtype, layout=dest.layout)
            ttnn.copy_host_to_device_tensor(host, dest)

        def set_context(context, start, valid):
            stage([[start, valid, 0, 0, 0, 0, 0, 0]], context.meta)

        def run_context(context, taps):
            positions.rotate(context.meta, context.cos, context.sin)
            context.extend(taps)

        def draft():
            positions.prepare(block, anchor, position, model.embed_table)
            base, recurrent = projection.project(block.forward(), last)
            markov.score(base, recurrent, anchor)

        prefill.run_chunk([42, 43], 0)
        set_context(prompt_context, 0, 2)
        run_context(prompt_context, prefill.taps)
        model.run(44, 2)
        set_context(decode_context, 2, 1)
        run_context(decode_context, decode_taps)
        stage([[45]], anchor)
        stage([3], position)
        draft()
        ttnn.synchronize_device(device)
        model.capture()

        def capture(body):
            trace = ttnn.begin_trace_capture(device, cq_id=0)
            body()
            ttnn.end_trace_capture(device, trace, cq_id=0)
            return trace

        traces = [
            capture(lambda: run_context(prompt_context, prefill.taps)),
            capture(lambda: run_context(decode_context, decode_taps)),
            capture(draft),
        ]

        def replay(i, *, blocking=False):
            ttnn.execute_trace(device, traces[i], cq_id=0, blocking=blocking)

        try:
            for case, row in enumerate(data[:2]):
                prompt = row["ids"].tolist()
                model.reset()
                logits, pos = prefill.prefill(prompt)
                first = int(logits.argmax())
                baseline = model.greedy(first, pos, steps)
                model.reset()
                logits = prefill.run_chunk(prompt, 0)
                assert int(logits.argmax()) == first
                set_context(prompt_context, 0, pos)
                replay(0)
                current = first
                emitted, timings, histogram = [], [], Counter()
                while len(emitted) < steps:
                    stage([[current]], anchor)
                    stage([pos], position)
                    ttnn.synchronize_device(device)
                    t0 = time.perf_counter()
                    replay(2, blocking=True)
                    timings.append((time.perf_counter() - t0) * 1e3)
                    proposed = [ttnn.to_torch(idx).item() for idx in markov.ids]
                    matched = 0
                    for candidate in [*proposed, None]:
                        current = model.step_greedy(current, pos)
                        set_context(decode_context, pos, 1)
                        replay(1)
                        pos += 1
                        emitted.append(current)
                        matched += current == candidate
                        if len(emitted) == steps or current != candidate:
                            break
                    histogram[matched] += 1
                total = sum(histogram.values())
                assert emitted == baseline
                print(
                    f"prompt {case} ({len(prompt)} input tokens): {steps} IDs match serial B1, "
                    f"{total} rounds, {steps / total:.3f} emitted/round, "
                    f"accepted histogram {[histogram[i] for i in range(proposals + 1)]}, "
                    f"draft median {statistics.median(timings):.2f} ms, "
                    f"p95 {statistics.quantiles(timings, n=20, method='inclusive')[-1]:.2f} ms",
                    flush=True,
                )
        finally:
            for trace in traces:
                ttnn.release_trace(device, trace)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=64)
    main(parser.parse_args().steps)
