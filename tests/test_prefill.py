"""Chunked prefill + traced decode vs HF capture."""

import sys
import time

import torch
import ttnn

from inkling_tt import paths
from inkling_tt.model import Precision
from inkling_tt.model2 import InklingTT2
from inkling_tt.prefill import Prefill


def main():
    chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    cap = torch.load(paths.golden("capture_haiku.pt"), weights_only=False)
    ids, gen, ref = cap["ids"][0].tolist(), cap["gen"], cap["logits"]
    dev = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=400_000_000)
    prec = Precision(
        expert_dtype=ttnn.bfloat8_b,
        dense_dtype=ttnn.bfloat16,
        expert_fidelity=ttnn.MathFidelity.HiFi4,
        dense_fidelity=ttnn.MathFidelity.HiFi4,
    )
    prec.experts = paths.experts("stream")
    print(f"settings: experts={prec.experts} chunk={chunk} lmax=2048", flush=True)
    m = InklingTT2(dev, paths.hf_sd(), prec, lmax=2048, log=lambda s: None)
    pf = Prefill(m, chunk)
    m.capture()
    for _ in range(2):
        m.reset()
        t0 = time.time()
        lg, pos = pf.prefill(ids)
        ttnn.synchronize_device(dev)
        print(f"prefill {len(ids)} tok: {time.time() - t0:.2f}s", flush=True)
    r = ref[0]
    print(f"prefill logits relerr {((lg - r).norm() / r.norm()).item():.3e} tt {int(lg.argmax())} hf {int(r.argmax())}")
    for k, tok in enumerate(gen):
        lg = m.step(tok, pos + k)
        r = ref[k + 1]
        print(
            f"decode {k} relerr {((lg - r).norm() / r.norm()).item():.3e} tt {int(lg.argmax())} hf {int(r.argmax())}",
            flush=True,
        )
    ttnn.close_device(dev)


main()
