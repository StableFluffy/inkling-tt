"""Teacher-forced agreement vs HF greedy (eval3x128.pt) + free-run divergence.

Args: expert_dtype dense_dtype head_dtype
"""

import json
import sys
import time

import torch
import ttnn

from inkling_tt import paths
from inkling_tt.device import open_device
from inkling_tt.model import Precision
from inkling_tt.model2 import InklingTT2
from inkling_tt.prefill import Prefill


def main():
    ed, dd, hd = (
        getattr(ttnn, a) for a in (sys.argv[1:4] if len(sys.argv) > 3 else ("bfloat8_b", "bfloat16", "bfloat16"))
    )
    fid = getattr(ttnn.MathFidelity, sys.argv[4]) if len(sys.argv) > 4 else ttnn.MathFidelity.HiFi4
    data = torch.load(paths.golden("eval3x128.pt"), weights_only=False)
    dev = open_device()
    prec = Precision(expert_dtype=ed, dense_dtype=dd, expert_fidelity=fid, dense_fidelity=ttnn.MathFidelity.HiFi4)
    prec.head_dtype = hd
    import os

    prec.experts = paths.experts("stream")
    print(f"settings: experts={prec.experts} lmax=2048 chunk=128", flush=True)
    prec.device_argmax = os.environ.get("INK_ARGMAX", "0") == "1"
    m = InklingTT2(dev, paths.hf_sd(), prec, lmax=2048, log=lambda s: None)
    pf = Prefill(m, 128)
    m.capture()
    tot = dict(n=0, top1=0, top5=0, rel=0.0)
    res = []
    for d in data:
        ids, gen, ref = d["ids"].tolist(), d["gen"], d["logits"].float()
        m.reset()
        t0 = time.time()
        lg, pos = pf.prefill(ids)
        ttft = time.time() - t0
        n1 = n5 = 0
        rels = []
        steps = []
        for k in range(len(gen)):
            r = ref[k]
            top = int(lg.argmax())
            n1 += top == int(r.argmax())
            n5 += top in set(r.topk(5).indices.tolist())
            rels.append(((lg - r).norm() / r.norm()).item())
            if k + 1 < len(gen):
                t0 = time.time()
                lg = m.step(gen[k], pos + k)
                steps.append(time.time() - t0)
        # free run greedy
        m.reset()
        lg, pos = pf.prefill(ids)
        fr = []
        for k in range(len(gen)):
            t = int(lg.argmax())
            fr.append(t)
            if k + 1 < len(gen):
                lg = m.step(t, pos + k)
        div = next((k for k in range(len(gen)) if fr[k] != gen[k]), len(gen))
        n = len(gen)
        st = sorted(steps)[len(steps) // 2]
        res.append(
            dict(
                prompt_len=len(ids),
                n=n,
                top1=n1 / n,
                top5=n5 / n,
                rel_mean=sum(rels) / n,
                diverge_at=div,
                ttft_s=ttft,
                step_ms=st * 1e3,
            )
        )
        print(json.dumps(res[-1]), flush=True)
        tot["n"] += n
        tot["top1"] += n1
        tot["top5"] += n5
        tot["rel"] += sum(rels)
    print(
        f"TOTAL top1 {tot['top1'] / tot['n']:.4f} top5 {tot['top5'] / tot['n']:.4f} rel {tot['rel'] / tot['n']:.4f}  "
        f"cfg {sys.argv[1:]}",
        flush=True,
    )
    ttnn.close_device(dev)
    if len(sys.argv) <= 1 or sys.argv[1:5] == ["bfloat8_b", "bfloat16", "bfloat16", "HiFi4"]:
        assert tot["top1"] >= 0.97 * tot["n"] and tot["top5"] >= 0.99 * tot["n"] and tot["rel"] <= 0.05 * tot["n"], tot
        print("PASS", flush=True)


main()
