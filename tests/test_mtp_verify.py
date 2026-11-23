import argparse
import statistics
import time

import torch
import ttnn

from inkling_tt import paths
from inkling_tt.model import VOCAB
from inkling_tt.model2 import InklingTT2
from inkling_tt.mtp_target import TargetVerify
from inkling_tt.prefill import Prefill
from serve.runtime import DEVICE_PARAMS, default_precision


def main(trace_only):
    dev = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    try:
        lmax = 1056
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

        def serial(start):
            rows, taps, ids = [], [], []
            for i, token in enumerate(block):
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
        for pos in (len(prompt), 1023, 1024, lmax - len(block)):
            seed(pos)
            expected = references[pos] = serial(pos)
            seed(pos)
            verifier.stage(block, pos)
            verifier.verify()
            check_outputs(expected)
            print(f"position {pos}: four rows match serial B1 logits and taps", flush=True)
            if pos == len(prompt):
                verifier.shared = False
                verifier.verify()
                check_outputs(expected)
                verifier.shared = True
            if trace_only:
                continue
            for rows in range(1, len(block) + 1):
                seed(pos)
                for r in range(rows):
                    model.run(block[r], pos + r)
                states = [
                    [ttnn.to_torch(t) for t in (visit.k, visit.v, *visit.conv)] for visit in model.visits.values()
                ]
                seed(pos)
                meta = torch.tensor([[pos, rows, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
                host = ttnn.from_torch(meta, dtype=ttnn.uint32)
                ttnn.copy_host_to_device_tensor(host, verifier.commit.meta)
                verifier.commit.commit()
                for (slot, visit), expected_state in zip(model.visits.items(), states):
                    for i, (t, ref) in enumerate(zip((visit.k, visit.v, *visit.conv), expected_state)):
                        actual = ttnn.to_torch(t)
                        assert torch.equal(actual, ref), (pos, rows, slot, i, (actual != ref).sum().item())
            print(f"position {pos}: all committed prefixes match serial B1 state", flush=True)

        try:
            verifier.stage(block, lmax - len(block) + 1)
        except ValueError:
            print("verify refuses blocks that cross lmax", flush=True)
        else:
            raise AssertionError("verify accepted a block that crosses lmax")

        model.capture()
        trace = ttnn.begin_trace_capture(dev, cq_id=0)
        verifier.verify()
        ttnn.end_trace_capture(dev, trace, cq_id=0)
        verifier.shared = False
        baseline_trace = ttnn.begin_trace_capture(dev, cq_id=0)
        verifier.verify()
        ttnn.end_trace_capture(dev, baseline_trace, cq_id=0)
        verifier.shared = True
        try:
            for pos, expected in references.items():
                seed(pos)
                verifier.stage(block, pos)
                ttnn.execute_trace(dev, trace, cq_id=0, blocking=True)
                check_outputs(expected)
            print("four verify replays match the serial B1 logits and taps", flush=True)
            pos = len(prompt)
            times = [[], [], []]
            traces = (model.trace_id, baseline_trace, trace)
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
            for name, samples in zip(("B1", "verify serial projections", "verify shared projections"), times):
                warm = samples[2:]
                print(
                    f"{name}: {statistics.median(warm):.2f} ms median, "
                    f"{statistics.quantiles(warm, n=20, method='inclusive')[-1]:.2f} ms p95",
                    flush=True,
                )
        finally:
            ttnn.release_trace(dev, baseline_trace)
            ttnn.release_trace(dev, trace)
            pf.close()
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-only", action="store_true")
    main(parser.parse_args().trace_only)
