"""Fused router vs the stock projection + rowvm router: routes, gamma and trace timing."""

import time

import torch
import ttnn

from inkling_tt import dense, rowvm

FP, BF, TILE, RM = ttnn.float32, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
dev = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=50_000_000)
torch.manual_seed(0)
w = (torch.randn(128, 2304) * 0.02).bfloat16()
b = torch.randn(128) * 0.01
handle = dense.prepare_weight(dev, w.T.contiguous(), BF)
bias_tile = ttnn.from_torch(b.reshape(1, 1, 1, 128), dtype=FP, layout=TILE, device=dev)
bias_rm = ttnn.from_torch(b.reshape(1, 1, 1, 128), dtype=FP, layout=RM, device=dev, memory_config=DRAM)
mism = 0
gerr = []
for trial in range(20):
    xt = torch.randn(1, 1, 1, 2304).bfloat16()
    x = ttnn.from_torch(xt, dtype=BF, layout=TILE, device=dev, memory_config=L1)
    ids0, g0 = rowvm.router(dense.linear(x, handle), bias_tile)
    ids1, g1 = rowvm.fused_router(x, handle.tensor, bias_rm)
    i0, i1 = ttnn.to_torch(ids0).reshape(-1).tolist(), ttnn.to_torch(ids1).reshape(-1).tolist()
    a0, a1 = ttnn.to_torch(g0).reshape(-1)[:8].float(), ttnn.to_torch(g1).reshape(-1)[:8].float()
    lg = xt.double().reshape(1, -1) @ w.double().T
    s = torch.sigmoid(lg).reshape(-1)
    ref = (s + b.double()).topk(8).indices.sort().values.tolist()
    mism += i0 != i1
    gerr.append(((a0 - a1).abs().max() / a0.abs().max()).item())
    if trial < 3 or i0 != i1:
        print(
            f"trial {trial}: stock {i0} fused {i1} fp64 {ref} gamma0 {a0.tolist()[:3]} gamma1 {a1.tolist()[:3]}",
            flush=True,
        )
print(f"route mismatches {mism}/20, max gamma rel diff {max(gerr):.3e}", flush=True)


def stock():
    rowvm.router(dense.linear(x, handle), bias_tile)


def fused():
    rowvm.fused_router(x, handle.tensor, bias_rm)


for name, fn in (("stock", stock), ("fused", fused)):
    fn()
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(32):
        fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ts = []
    for _ in range(5):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        ts.append((time.perf_counter() - t0) / 32 * 1e6)
    ttnn.release_trace(dev, tid)
    print(f"{name}: {sorted(ts)[2]:.2f} us/call", flush=True)
ttnn.close_device(dev)
