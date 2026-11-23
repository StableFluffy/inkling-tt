"""300-token prompt: chunked prefill vs token-by-token decode path vs HF (long300.pt)."""

import os
import time

import torch
import ttnn

from inkling_tt import paths
from inkling_tt.model import Precision
from inkling_tt.model2 import InklingTT2
from inkling_tt.prefill import Prefill

g = torch.load(paths.golden("long300.pt"), weights_only=False)
ids, gen, ref = g["ids"].tolist(), g["gen"], g["logits"].float()
dev = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=400_000_000)
prec = Precision(
    expert_dtype=ttnn.bfloat8_b,
    dense_dtype=ttnn.bfloat16,
    expert_fidelity=ttnn.MathFidelity.HiFi4,
    dense_fidelity=ttnn.MathFidelity.HiFi4,
)
prec.experts = paths.experts("stream")
print(f"settings: experts={prec.experts} lmax=2048 chunk=128", flush=True)
prec.device_argmax = os.environ.get("INK_ARGMAX", "0") == "1"
m = InklingTT2(dev, paths.hf_sd(), prec, lmax=2048, log=lambda s: None)
pf = Prefill(m, 128)
m.capture()
rel = lambda a, b: ((a - b).norm() / b.norm()).item()


def tf(lg, pos, tag):
    out = [rel(lg, ref[0])]
    agree = int(lg.argmax()) == int(ref[0].argmax())
    for k, t in enumerate(gen[:-1]):
        lg = m.step(t, pos + k)
        out.append(rel(lg, ref[k + 1]))
        agree += int(lg.argmax()) == int(ref[k + 1].argmax())
    print(
        f"{tag}: prefill-logit relerr {out[0]:.3e}  decode relerr mean {sum(out[1:]) / len(out[1:]):.3e}  "
        f"argmax agree {agree}/{len(gen)}",
        flush=True,
    )
    assert out[0] < 0.04 and agree >= len(gen) - 2, (tag, out[0], agree)


for rep in range(2):
    m.reset()
    t0 = time.time()
    lg, pos = pf.prefill(ids)
    print(f"chunked prefill {time.time() - t0:.2f}s", flush=True)
    tf(lg, pos, f"chunked#{rep}")
m.reset()
t0 = time.time()
for p, t in enumerate(ids):
    lg = m.step(t, p)
print(f"token-by-token prefill {time.time() - t0:.2f}s", flush=True)
tf(lg, len(ids), "tokenwise")
ttnn.close_device(dev)
print("PASS", flush=True)
