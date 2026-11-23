"""Real decode loop timings: greedy (device argmax) and logits readback, changing tokens/positions."""

import os
import time

import torch
import ttnn

from inkling_tt import paths
from inkling_tt.device import open_device
from inkling_tt.model import Precision
from inkling_tt.model2 import InklingTT2
from inkling_tt.prefill import Prefill

dev = open_device()
prec = Precision(
    expert_dtype=ttnn.bfloat8_b,
    dense_dtype=ttnn.bfloat16,
    expert_fidelity=ttnn.MathFidelity.HiFi4,
    dense_fidelity=ttnn.MathFidelity.HiFi4,
)
prec.experts = paths.experts("stream")
print(
    f"settings: experts={prec.experts} lmax={os.environ.get('INK_LMAX', '2048')} "
    f"attn_decode={os.environ.get('INK_ATTN_DECODE', '1')} reps=64 (first 4 discarded) "
    "statistic=median,p95 of per-step wall time",
    flush=True,
)
m = InklingTT2(dev, paths.hf_sd(), prec, lmax=int(os.environ.get("INK_LMAX", "2048")), log=lambda s: None)
pf = Prefill(m, 128)
m.capture()
cap = torch.load(paths.golden("capture_haiku.pt"), weights_only=False)
ids = cap["ids"][0].tolist()
for mode in ("greedy", "logits"):
    m.reset()
    lg, pos = pf.prefill(ids)
    tok = int(lg.argmax())
    ts = []
    out = []
    for k in range(64):
        t0 = time.time()
        tok = m.step_greedy(tok, pos + k) if mode == "greedy" else int(m.step(tok, pos + k).argmax())
        ts.append((time.time() - t0) * 1e3)
        out.append(tok)
    s = sorted(ts[4:])
    print(
        f"{mode}: median {s[len(s) // 2]:.2f} ms p95 {s[int(len(s) * 0.95)]:.2f} -> {1000 / s[len(s) // 2]:.1f} tok/s; "
        f"first tokens {out[:8]} vs HF {cap['gen'][1:9]}",
        flush=True,
    )
ttnn.close_device(dev)
