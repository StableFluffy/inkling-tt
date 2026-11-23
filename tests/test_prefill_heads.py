"""Compare the fused head clip/RMS kernel with the TTNN reference sequence."""

from statistics import median
from time import perf_counter

import torch
import ttnn

import inkling_tt.prefill as pp
from inkling_tt.model import EPS, HD, LOG_ALPHA, LOG_FLOOR
from inkling_tt.model2 import BF, FP, TILE


def bench(dev, fn):
    fn()
    ttnn.synchronize_device(dev)
    times = []
    for _ in range(9):
        start = perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        times.append((perf_counter() - start) * 1e3)
    return median(times)


def real(dev, path):
    for case in torch.load(path, weights_only=False):
        host, heads, rms, thr, off = case["x"], case["heads"], case["rms"], case["thr"], case["col_offset"]
        t = host.shape[-2]
        x = ttnn.from_torch(host, dtype=BF, layout=TILE, device=dev)
        y = ttnn.slice(x, [0, 0, 0, off], [1, 1, t, off + heads * HD])
        y = ttnn.reshape(y, [1, t, heads, HD])
        y = pp.Prefill.hrms(y) if rms else pp.Prefill.hclip(y, thr)
        a = ttnn.to_torch(ttnn.permute(y, (0, 2, 1, 3))).float()
        b = ttnn.to_torch(pp._heads(x, off, heads, rms=rms, threshold=thr)).float()
        print("REAL", heads, rms, "mismatches", int((a != b).sum()), "of", a.numel(), flush=True)


def main():
    import sys

    torch.manual_seed(3)
    dev = ttnn.open_device(device_id=0, l1_small_size=16384)
    dev.enable_program_cache()
    if len(sys.argv) > 2:
        ttnn.close_device(dev)
        position(sys.argv[1])
        return
    if len(sys.argv) > 1:
        try:
            real(dev, sys.argv[1])
        finally:
            ttnn.close_device(dev)
        return
    try:
        for t in (128, 1024):
            pos = 8192
            tau = 1.0 + LOG_ALPHA * ((pos + 1 + torch.arange(t)) / LOG_FLOOR).clamp_min(1.0).log()
            tau4 = ttnn.from_torch(tau.reshape(1, t, 1, 1), dtype=FP, layout=TILE, device=dev)
            tau_col = ttnn.from_torch(
                tau.reshape(1, 1, t, 1).expand(1, 1, t, 32).contiguous(), dtype=FP, layout=TILE, device=dev
            )
            for heads, width, mode in ((24, 6560, "clip_tau"), (24, 6560, "clip"), (8, 1024, "clip"), (8, 1024, "rms")):
                host = (torch.randn(1, 1, t, width) * torch.rand(1, 1, t, 1) * 6).bfloat16()
                host[..., :3, :] = 0
                x = ttnn.from_torch(host, dtype=BF, layout=TILE, device=dev)

                def reference():
                    y = ttnn.slice(x, [0, 0, 0, 0], [1, 1, t, heads * HD])
                    y = ttnn.reshape(y, [1, t, heads, HD])
                    y = pp.Prefill.hrms(y) if mode == "rms" else pp.Prefill.hclip(y, 22.627416997969522)
                    if mode == "clip_tau":
                        y = ttnn.multiply(ttnn.typecast(y, FP), tau4, dtype=BF)
                    return ttnn.permute(y, (0, 2, 1, 3))

                def fused():
                    return pp._heads(
                        x,
                        0,
                        heads,
                        rms=mode == "rms",
                        threshold=22.627416997969522,
                        tau=tau_col if mode == "clip_tau" else None,
                    )

                a = ttnn.to_torch(reference()).float()
                b = ttnn.to_torch(fused()).float()
                ref = host[0, 0, :, : heads * HD].float().reshape(t, heads, HD)
                ss = ref.square().sum(-1, keepdim=True)
                exact = ref * (
                    torch.rsqrt(ss / HD + EPS)
                    if mode == "rms"
                    else (22.627416997969522 / (ss.sqrt() + EPS)).clamp(max=1.0)
                )
                exact = exact.bfloat16().float()
                if mode == "clip_tau":
                    exact = (exact * tau.reshape(t, 1, 1)).bfloat16().float()
                exact = exact.permute(1, 0, 2).unsqueeze(0)
                rel = lambda u: ((u - exact).norm() / exact.norm()).item()
                print(
                    "HEADS",
                    t,
                    mode,
                    "equal_frac",
                    (a == b).float().mean().item(),
                    "max_abs",
                    (a - b).abs().max().item(),
                    "rel_ref",
                    rel(a),
                    "rel_fused",
                    rel(b),
                    "ms_ref",
                    round(bench(dev, reference), 3),
                    "ms_fused",
                    round(bench(dev, fused), 3),
                    flush=True,
                )
    finally:
        ttnn.close_device(dev)


def position(path):
    """Place one mismatching head row at every token row and every head slot."""
    dev = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        case = torch.load(path, weights_only=False)[0]
        host, heads, thr, off = case["x"], case["heads"], case["thr"], case["col_offset"]
        t = host.shape[-2]
        x = ttnn.from_torch(host, dtype=BF, layout=TILE, device=dev)
        y = ttnn.reshape(ttnn.slice(x, [0, 0, 0, off], [1, 1, t, off + heads * HD]), [1, t, heads, HD])
        a = ttnn.to_torch(ttnn.permute(pp.Prefill.hclip(y, thr), (0, 2, 1, 3))).float()
        b = ttnn.to_torch(pp._heads(x, off, heads, threshold=thr)).float()
        _, h, tok, _ = (a != b).nonzero()[0].tolist()
        row = host[0, 0, tok, off + h * HD : off + (h + 1) * HD]
        grid = row.repeat(1, 1, 32, heads).bfloat16()
        g = ttnn.from_torch(grid, dtype=BF, layout=TILE, device=dev)
        ref = ttnn.to_torch(pp.Prefill.hclip(ttnn.reshape(g, [1, 32, heads, HD]), thr)).float()[0]
        got = ttnn.to_torch(pp._heads(g, 0, heads, threshold=thr)).float()[0]
        base = ref[0, 0]
        print(
            "POS ref rows differ from row0/head0:",
            [(i, j) for i in range(32) for j in range(heads) if not torch.equal(ref[i, j], base)][:20],
            flush=True,
        )
        print("POS fused rows differ from ref:", [i for i in range(32) if not torch.equal(got[0, i], base)], flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
