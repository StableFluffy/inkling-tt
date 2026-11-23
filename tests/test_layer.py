"""model2 single layers token by token vs HF prefill captures."""

import sys

import torch
import ttnn

from inkling_tt import paths
from inkling_tt.model import GLOBAL_LAYERS, Precision, Weights
from inkling_tt.model2 import BF, L1, RM, Ctx, E, Layer, Ops, Step, Visit


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def main():
    layers = [tuple(int(y) for y in (x + ":0").split(":")[:2]) for x in sys.argv[1].split(",")]
    cap = torch.load(paths.golden("capture_haiku.pt"), weights_only=False)
    st0 = cap["steps"][0]
    dev = ttnn.open_device(device_id=0, l1_small_size=16384)
    prec = Precision(
        expert_dtype=ttnn.bfloat8_b,
        dense_dtype=ttnn.bfloat16,
        expert_fidelity=ttnn.MathFidelity.HiFi4,
        dense_fidelity=ttnn.MathFidelity.HiFi4,
    )
    c = Ctx(dev, prec)
    ops = Ops(c)
    ops.dummy = c.up(torch.zeros(1, 1, 1, E), BF, RM, L1)
    wts = Weights(paths.hf_sd())
    for li, pidx in layers:
        L = Layer(c, ops, wts, li)
        rec = st0[f"layer{li}"][pidx]
        x_in, bs = rec["args"][0], rec["args"][1]
        mref = rec["args"][2] if len(rec["args"]) > 2 else None
        T = x_in.shape[1]
        st = Step(c, 2048)
        vs = Visit(c, li in GLOBAL_LAYERS, 2048)
        errs = []
        for t in range(T):
            st.write(x_in[0, t], t)
            bank = ops.bank0
            for j in range(bs.shape[2]):
                bank = ops.bank_append(bank, c.up(bs[0, t, j].float().reshape(1, 1, 1, -1)))
            msgs = (
                None
                if mref is None
                else [c.up(mref[0, t, q].float().reshape(1, 1, 1, -1), ttnn.float32) for q in range(4)]
            )
            y = L.forward(st.x, bank, vs, st, msgs)
            errs.append(rel(ttnn.to_torch(y).reshape(-1).float(), rec["out"][0, t].float()))
        print(
            f"layer {li} pass {pidx}: relerr max {max(errs):.2e} mean {sum(errs) / len(errs):.2e} "
            f"first {errs[0]:.2e} last {errs[-1]:.2e}",
            flush=True,
        )
    ttnn.close_device(dev)


main()
