"""Grouped message down projections (4608 -> 64, BF16) vs the tuned stock matmuls."""

import time

import torch
import ttnn

from inkling_tt import dense, rowvm

BF, TILE = ttnn.bfloat16, ttnn.TILE_LAYOUT
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
dev = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=50_000_000)
torch.manual_seed(0)
G = 3
ws = [(torch.randn(4608, 64) * 0.02).bfloat16() for _ in range(G)]
handles = [dense.prepare_weight(dev, w, BF) for w in ws]
inter = [ttnn.from_torch(w.reshape(1, 1, 4608, 64), dtype=BF, layout=TILE, device=dev, memory_config=DRAM) for w in ws]
xs_t = [torch.randn(1, 1, 1, 4608).bfloat16() for _ in range(G)]
xs = [ttnn.from_torch(x, dtype=BF, layout=TILE, device=dev, memory_config=L1) for x in xs_t]
xs_sh = [ttnn.to_memory_config(x, h.input_shards) for x, h in zip(xs, handles)]


def stock():
    return [
        ttnn.linear(
            x,
            h.tensor,
            program_config=h.program,
            compute_kernel_config=h.compute,
            memory_config=h.output_shards,
            dtype=BF,
        )
        for x, h in zip(xs_sh, handles)
    ]


def fused():
    return rowvm.matvec_group(zip(xs, inter))


a, b = stock(), fused()
for g in range(G):
    ta, tb = ttnn.to_torch(a[g]).float().reshape(-1)[:64], ttnn.to_torch(b[g]).float().reshape(-1)[:64]
    ref = (xs_t[g].double().reshape(1, -1) @ ws[g].double()).reshape(-1)
    print(
        f"group {g}: stock-vs-fused max ulp-ish {((ta - tb).abs() / ta.abs().clamp_min(1e-6)).max():.3e} "
        f"equal {int((ta == tb).sum())}/64  err stock {((ta - ref).norm() / ref.norm()):.3e} "
        f"fused {((tb - ref).norm() / ref.norm()):.3e}",
        flush=True,
    )
for name, fn in (("stock", stock), ("fused", fused)):
    fn()
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(16):
        fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ts = []
    for _ in range(5):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        ts.append((time.perf_counter() - t0) / 16 * 1e6)
    ttnn.release_trace(dev, tid)
    print(f"{name}: {sorted(ts)[2]:.2f} us per {G} projections", flush=True)
ttnn.close_device(dev)
