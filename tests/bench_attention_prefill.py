"""Time the tiled prefill attention kernel at representative chunk positions.

usage: bench_attention_prefill.py [rows]   env ATTN_VARIANTS=base:,project:PROJECT=1
"""

import os
import sys
from statistics import median
from time import perf_counter

import torch
import ttnn

import inkling_tt.prefill as pp
from inkling_tt.model2 import BF, DRAM, FP, RM, TILE


def variants():
    out = []
    for item in os.getenv("ATTN_VARIANTS", "base:").split(","):
        name, _, defs = item.partition(":")
        out.append((name, tuple(tuple(d.split("=")) for d in defs.split("+") if d)))
    return out


def main():
    torch.manual_seed(11)
    dev = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=20_000_000)
    dev.enable_program_cache()
    up = lambda x, dt=BF, layout=TILE: ttnn.from_torch(x.contiguous(), dtype=dt, layout=layout, device=dev)
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    try:
        capacity = 32768
        caches = {
            True: (up(torch.randn(1, 8, capacity, 128)), up(torch.randn(1, 8, capacity, 128))),
            False: (up(torch.randn(1, 8, 1024, 128)), up(torch.randn(1, 8, 1024, 128))),
        }
        q = up(torch.randn(1, 24, t, 128))
        k = up(torch.randn(1, 8, t, 128))
        v = up(torch.randn(1, 8, t, 128))
        sink = up(torch.randn(1, 1, t, 32))
        rel = torch.zeros(1, 24, t, 32)
        rel[..., :16] = torch.randn(1, 24, t, 16)
        relative = up(rel)
        table = up(torch.randn(1, 1, 32, 1024) * 0.1)
        exact = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        distance = ttnn.matmul(relative, table, compute_kernel_config=exact, dtype=FP)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, t, 3072)), BF, TILE, dev, DRAM)
        for is_global, start in ((False, 0), (False, 4096), (True, 0), (True, 8192 - t), (True, 32768 - t)):
            meta = up(torch.tensor([[start, t, 0, 0, 0, 0, 0, 0]], dtype=torch.int32), ttnn.uint32, RM)
            cache_k, cache_v = caches[is_global]
            reference = None
            for name, defs in variants():
                project = dict(defs).get("PROJECT") == "1"

                def run():
                    pp._tiled_attention(
                        q,
                        k,
                        v,
                        cache_k,
                        cache_v,
                        sink,
                        relative if project else distance,
                        meta,
                        is_global,
                        out,
                        retain_scores=True,
                        global_score_tiles=97,
                        table=table if project else None,
                    )

                run()
                result = ttnn.to_torch(out).clone()
                if reference is None:
                    reference = result
                times = []
                for _ in range(5):
                    begin = perf_counter()
                    run()
                    ttnn.synchronize_device(dev)
                    times.append((perf_counter() - begin) * 1e3)
                print(
                    "ATTN",
                    t,
                    "global" if is_global else "local",
                    start,
                    name,
                    "ms",
                    round(median(times), 3),
                    "bitwise",
                    torch.equal(result, reference),
                    flush=True,
                )
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
