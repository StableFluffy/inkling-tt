"""M2 full model vs HF capture; mode eager|trace."""

import sys
import time

import torch
import ttnn

from inkling_tt import paths
from inkling_tt.device import open_device
from inkling_tt.model import Precision
from inkling_tt.model2 import InklingTT2


def main():
    mode = sys.argv[1]
    layers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    cap = torch.load(paths.golden("capture_haiku.pt"), weights_only=False)
    ids, gen, ref = cap["ids"][0].tolist(), cap["gen"], cap["logits"]
    dev = open_device()
    prec = Precision(
        expert_dtype=ttnn.bfloat8_b,
        dense_dtype=ttnn.bfloat16,
        expert_fidelity=ttnn.MathFidelity.HiFi4,
        dense_fidelity=ttnn.MathFidelity.HiFi4,
    )
    t0 = time.time()
    import os

    prec.experts = paths.experts("stream")
    print(f"settings: experts={prec.experts} mode={mode}", flush=True)
    prec.device_argmax = os.environ.get("INK_ARGMAX", "0") == "1"
    m = InklingTT2(
        dev,
        paths.hf_sd(),
        prec,
        lmax=int(__import__("os").environ.get("INK_LMAX", "2048")),
        layers=layers,
        log=lambda s: None,
    )
    print(f"loaded {time.time() - t0:.0f}s", flush=True)
    if mode == "trace":
        t0 = time.time()
        m.capture()
        print(f"captured {time.time() - t0:.0f}s", flush=True)
    toks = ids + gen
    n_prompt = len(ids)
    times = []
    diffs = checked = 0
    for p, tok in enumerate(toks):
        t0 = time.time()
        lg = m.step(tok, p)
        dt = time.time() - t0
        times.append(dt)
        if p >= n_prompt - 1 and layers is None:
            k = p - (n_prompt - 1)
            r = ref[k]
            rl = ((lg - r).norm() / r.norm()).item()
            top, rtop = int(lg.argmax()), int(r.argmax())
            print(
                f"pos {p} {dt * 1000:.1f} ms relerr {rl:.3e} tt {top} hf {rtop} {'OK' if top == rtop else 'DIFF'}",
                flush=True,
            )
            checked += 1
            diffs += top != rtop
        elif p % 8 == 0:
            print(f"pos {p} {dt * 1000:.1f} ms", flush=True)
    st = sorted(times[3:])
    print(f"median step {st[len(st) // 2] * 1000:.1f} ms", flush=True)
    ttnn.close_device(dev)
    if layers is None:
        print(f"argmax agree {checked - diffs}/{checked}", flush=True)
        assert diffs <= 1, f"{diffs} of {checked} argmax decisions differ from HF"
        print("PASS", flush=True)


main()
