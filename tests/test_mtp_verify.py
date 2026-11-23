import argparse
import statistics
import time

import torch
import ttnn

from inkling_tt import paths, rowvm
from inkling_tt.model import VOCAB
from inkling_tt.model2 import InklingTT2
from inkling_tt.mtp_target import TargetVerify
from inkling_tt.prefill import Prefill
from serve.runtime import DEVICE_PARAMS, default_precision


def main(trace_only):
    dev = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    try:
        lmax = 1120
        model = InklingTT2(dev, paths.hf_sd(), default_precision(), lmax=lmax, capture_taps=True)
        pf = Prefill(model, 128)
        verifier = TargetVerify(model, 3, streaming=True, compare_serial=True)
        fixture = torch.load(paths.golden("capture_haiku.pt"), weights_only=False)
        prompt = fixture["ids"][0].tolist()
        block = fixture["gen"][:4]

        def seed(position):
            model.reset()
            prefix = (prompt * ((position + len(prompt) - 1) // len(prompt)))[:position]
            pf.prefill(prefix)

        def serial(start, tokens):
            rows, taps, ids = [], [], []
            for i, token in enumerate(tokens):
                model.run(token, start + i)
                rows.append(ttnn.to_torch(model.logits).flatten()[:VOCAB])
                taps.append(torch.stack([ttnn.to_torch(t).flatten() for t in model.taps]))
                ids.append(ttnn.to_torch(model.next_id).item())
            return torch.stack(rows), torch.stack(taps), ids

        def check_outputs(expected):
            expected_logits, expected_taps, expected_ids = expected
            actual = torch.stack([ttnn.to_torch(row).flatten()[:VOCAB] for row in verifier.logits])
            assert torch.equal(actual, expected_logits), (actual != expected_logits).sum(-1).tolist()
            assert [ttnn.to_torch(index).item() for index in verifier.ids] == expected_ids
            taps = torch.stack([ttnn.to_torch(t)[0, 0, : len(block)] for t in verifier.taps], dim=1)
            assert torch.equal(taps, expected_taps)

        references = {}
        for pos in (len(prompt), 1023, 1024, 1087, 1088, lmax - len(block)):
            seed(pos)
            expected = references[pos] = serial(pos, block)
            seed(pos)
            verifier.stage(block, pos)
            verifier.verify()
            check_outputs(expected)
            print(f"position {pos}: four rows match serial B1 logits and taps", flush=True)
            if pos == len(prompt):
                seed(pos)
                verifier.stage(block, pos)
                verifier.shared = False
                verifier.verify()
                check_outputs(expected)
                verifier.shared = True

        try:
            verifier.stage(block, lmax - len(block) + 1)
        except ValueError:
            print("verify refuses blocks that cross lmax", flush=True)
        else:
            raise AssertionError("verify accepted a block that crosses lmax")

        model.capture()
        verify_traces = []
        for glue, attention in ((False, False), (True, False), (True, True)):
            seed(len(prompt))
            verifier.stage(block, len(prompt))
            verifier.verify(grouped_glue=glue, grouped_attention=attention)
            check_outputs(references[len(prompt)])
            ttnn.synchronize_device(dev)
            seed(len(prompt))
            verifier.stage(block, len(prompt))
            rowvm._group_program.cache_clear()
            trace = ttnn.begin_trace_capture(dev, cq_id=0)
            verifier.verify(grouped_glue=glue, grouped_attention=attention)
            ttnn.end_trace_capture(dev, trace, cq_id=0)
            verify_traces.append(trace)
        trace = verify_traces[-1]
        try:
            for pos, expected in references.items():
                seed(pos)
                verifier.stage(block, pos)
                ttnn.execute_trace(dev, trace, cq_id=0, blocking=True)
                check_outputs(expected)
                seed(pos)
                verifier.stage(block, pos)
                ttnn.execute_trace(dev, verify_traces[1], cq_id=0, blocking=True)
                check_outputs(expected)
                if trace_only:
                    continue
                for rows in range(1, len(block)):
                    alternative = prompt[: len(block) - rows]
                    if alternative[0] == block[rows]:
                        alternative = [(alternative[0] + 1) % VOCAB, *alternative[1:]]
                    seed(pos)
                    serial(pos, block[:rows])
                    oracle = serial(pos + rows, alternative)
                    seed(pos)
                    verifier.stage(block, pos)
                    ttnn.execute_trace(dev, trace, cq_id=0, blocking=True)
                    actual = serial(pos + rows, alternative)
                    assert all(torch.equal(a, b) for a, b in zip(actual[:2], oracle[:2])), (pos, rows)
                    assert actual[2] == oracle[2], (pos, rows)
                print(f"position {pos}: every rejected suffix can be overwritten by B1", flush=True)
            print(f"{len(references)} verify replays match the serial B1 logits and taps", flush=True)
            pos = len(prompt)
            traces = (model.trace_id, *verify_traces)
            times = [[] for _ in traces]
            for trial in range(12):
                for kind in ((trial + i) % len(traces) for i in range(len(traces))):
                    seed(pos)
                    if kind:
                        verifier.stage(block, pos)
                    else:
                        model.st.write(model.embed_row(block[0]), pos)
                    ttnn.synchronize_device(dev)
                    start = time.perf_counter()
                    ttnn.execute_trace(dev, traces[kind], cq_id=0, blocking=True)
                    times[kind].append(1000 * (time.perf_counter() - start))
            for name, samples in zip(
                ("B1", "verify serial epilogues", "verify grouped epilogues", "verify grouped attention"), times
            ):
                warm = samples[2:]
                print(
                    f"{name}: {statistics.median(warm):.2f} ms median, "
                    f"{statistics.quantiles(warm, n=20, method='inclusive')[-1]:.2f} ms p95",
                    flush=True,
                )
        finally:
            for trace in verify_traces:
                ttnn.release_trace(dev, trace)
            pf.close()
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-only", action="store_true")
    main(parser.parse_args().trace_only)
