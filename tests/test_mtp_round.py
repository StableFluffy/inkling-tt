import argparse
import statistics
import time

import torch
import ttnn

from inkling_tt import paths
from inkling_tt.model2 import InklingTT2
from inkling_tt.mtp_round import SpeculativeRound
from inkling_tt.prefill import Prefill
from serve.runtime import DEVICE_PARAMS, default_precision


def main(proposals):
    dev = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    try:
        lmax = 1056
        model = InklingTT2(dev, paths.hf_sd(), default_precision(), lmax=lmax, capture_taps=True)
        pf = Prefill(model, 128)
        spec = SpeculativeRound(model, [pf], paths.snapshot(), proposals=proposals)
        spec.warm()
        model.capture()
        spec.capture()
        try:
            data = torch.load(paths.golden("eval3x128.pt"), weights_only=False)
            prompts = [(name, row["ids"].tolist()) for name, row in zip(("chat", "code", "Korean"), data)]
            base = prompts[0][1]
            prompts.append(("ring wrap", (base * (1023 // len(base) + 1))[:1023]))
            for name, prompt in prompts:
                count = 32
                expected = None
                samples = [[], []]
                for i in range(4):
                    for mode in (0, 1) if i % 2 == 0 else (1, 0):
                        model.reset()
                        for offset in range(0, len(prompt), pf.chunk):
                            chunk = prompt[offset : offset + pf.chunk]
                            logits = pf.run_chunk(chunk, offset, return_logits=offset + len(chunk) == len(prompt))
                            if mode:
                                spec.append_prompt(pf)
                        current, pos = int(logits.argmax()), len(prompt)
                        emitted = []
                        ttnn.synchronize_device(dev)
                        start = time.perf_counter()
                        while len(emitted) < count:
                            if mode and spec.can_run(pos) and count - len(emitted) >= spec.rows:
                                ids, pos = spec.run(current, pos)
                            else:
                                ids, pos = [model.step_greedy(current, pos)], pos + 1
                            emitted.extend(ids)
                            current = ids[-1]
                        samples[mode].append(1000 * (time.perf_counter() - start) / count)
                        if expected is None:
                            expected = emitted
                        else:
                            assert emitted == expected, (
                                name,
                                mode,
                                next(i for i, (a, b) in enumerate(zip(emitted, expected)) if a != b),
                            )
                for label, timings in zip(("B1", "MTP"), samples):
                    p95 = statistics.quantiles(timings, n=20, method="inclusive")[-1]
                    print(
                        f"{name}: {label} {count} identical IDs, "
                        f"{statistics.median(timings):.2f} ms/ID median, {p95:.2f} ms/ID p95",
                        flush=True,
                    )
        finally:
            spec.close()
            pf.close()
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=int, choices=range(1, 8), default=3)
    main(parser.parse_args().proposals)
