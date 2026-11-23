"""Two resident prefill buckets (128 and 2048 rows) with decode captured after both.

usage: test_prefill_buckets.py   env PREFILL_LMAX, PREFILL_REPS, PREFILL_LENGTHS
TTFT rows: warm, from position 0, embedding upload + traced prefill + final head + logits readback + host argmax,
median of PREFILL_REPS. Fails on non-finite handoff logits, golden drift or bucket mismatch.
"""

import os
import statistics
import time

import torch
import ttnn

from inkling_tt import paths
from inkling_tt.device import open_device
from inkling_tt.model import Precision
from inkling_tt.model2 import InklingTT2
from inkling_tt.prefill import Prefill, prefill_buckets


def main():
    dev = open_device()
    spec = Precision(
        expert_dtype=ttnn.bfloat8_b,
        dense_dtype=ttnn.bfloat16,
        expert_fidelity=ttnn.MathFidelity.HiFi4,
        dense_fidelity=ttnn.MathFidelity.HiFi4,
    )
    spec.experts = paths.experts("stream")
    print(
        "SETTINGS experts",
        spec.experts,
        "lmax",
        os.getenv("PREFILL_LMAX", "32768"),
        "weights",
        paths.hf_sd(),
        flush=True,
    )
    m = InklingTT2(dev, paths.hf_sd(), spec, lmax=int(os.getenv("PREFILL_LMAX", "32768")), log=lambda _: None)
    try:
        t = time.perf_counter()
        short, long = Prefill(m, 128), Prefill(m, 2048)
        print("PREFILL_INIT", time.perf_counter() - t, flush=True)
        t = time.perf_counter()
        m.capture()
        print(
            "CAPTURE",
            time.perf_counter() - t,
            "decode",
            m.trace_id,
            "short",
            short.body_trace,
            "long",
            long.body_trace,
            flush=True,
        )
        for name in ("DRAM", "L1", "TRACE"):
            mem = ttnn.get_memory_view(dev, getattr(ttnn.BufferType, name))
            print(
                "MEM",
                name,
                mem.num_banks,
                mem.total_bytes_per_bank,
                mem.total_bytes_allocated_per_bank,
                mem.largest_contiguous_bytes_free_per_bank,
                flush=True,
            )
        buckets = (short, long)
        g = torch.load(paths.golden("long300.pt"), weights_only=False)
        source = g["ids"].tolist()
        refs = g["logits"].float()
        lm, pos = prefill_buckets(buckets, source)
        rel, agree = [], 0
        for i, r in enumerate(refs):
            rel.append(((lm - r).norm() / r.norm()).item())
            agree += int(lm.argmax()) == int(r.argmax())
            if i + 1 < len(refs):
                lm = m.step(g["gen"][i], pos + i)
        print(
            "GOLDEN",
            len(source),
            "rel",
            rel[0],
            "continuation",
            sum(rel[1:]) / len(rel[1:]),
            "agree",
            agree,
            len(refs),
            flush=True,
        )

        def ids(n):
            return (source * (n // len(source) + 1))[:n]

        mixed, _ = prefill_buckets(buckets, ids(2100))
        small, _ = short.prefill(ids(2100))
        print("MIXED 2100 equals 128-row chunks", torch.equal(mixed, small), flush=True)
        wide, _ = long.prefill(ids(4100))
        narrow, _ = short.prefill(ids(4100))
        print(
            "LONG 4100 2048-row equals 128-row chunks",
            torch.equal(wide, narrow),
            "rel",
            ((wide - narrow).norm() / narrow.norm()).item(),
            flush=True,
        )
        lengths = [
            n for n in map(int, os.getenv("PREFILL_LENGTHS", "128,300,1024,2048,8192").split(",")) if n <= m.lmax
        ]
        for n in lengths:
            times = []
            for _ in range(int(os.getenv("PREFILL_REPS", "3"))):
                ttnn.synchronize_device(dev)
                t = time.perf_counter()
                lg, pos = prefill_buckets(buckets, ids(n))
                tok = int(lg.argmax())
                times.append(time.perf_counter() - t)
            median = statistics.median(times)
            print(
                "TTFT",
                n,
                f"median={median:.6f}",
                f"tok_s={n / median:.1f}",
                f"samples={times}",
                f"argmax={tok}",
                flush=True,
            )
        # A prompt of exactly lmax tokens is a full-cache prefill measurement with no decode room left.
        # The decode handoff runs on the longest swept prompt that leaves `steps` positions in the cache.
        steps = 16
        n = max([k for k in lengths if k + steps <= m.lmax] or [128])
        lg, pos = prefill_buckets(buckets, ids(n))
        finite, out = bool(lg.isfinite().all()), []
        for k in range(steps):
            tok = int(lg.argmax())
            out.append(tok)
            lg = m.step(tok, pos + k)
            finite &= bool(lg.isfinite().all())
        print("HANDOFF", n, "positions", pos, pos + steps - 1, "finite", finite, "tokens", out, flush=True)
        assert finite, "non-finite decode logits after prefill"
        assert rel[0] < 0.04 and agree >= len(refs) - 2, (rel[0], agree)
        assert torch.equal(mixed, small) and torch.equal(wide, narrow), "bucket results differ from 128-row chunks"
        print("PASS", flush=True)
        long.close()
        short.close()
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
