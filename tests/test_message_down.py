"""Fused controller input + down projection vs message_input + matvec_group: values and timing."""

import time

import torch
import ttnn

from inkling_tt import rowvm

FP, BF, TILE = ttnn.float32, ttnn.bfloat16, ttnn.TILE_LAYOUT
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
dev = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=50_000_000)
torch.manual_seed(0)
for G in (3, 1):
    us = [torch.randn(1, 1, 1, 2304) * 2 for _ in range(G)]
    ms = [torch.randn(1, 1, 1, 2304) for _ in range(G)]
    ws = [(torch.randn(4608, 64) * 0.02).bfloat16() for _ in range(G)]
    u = [ttnn.from_torch(t, dtype=FP, layout=TILE, device=dev, memory_config=L1) for t in us]
    m = [ttnn.from_torch(t, dtype=FP, layout=TILE, device=dev, memory_config=L1) for t in ms]
    w = [ttnn.from_torch(t.reshape(1, 1, 4608, 64), dtype=BF, layout=TILE, device=dev, memory_config=DRAM) for t in ws]

    def current():
        ins = rowvm.qkv_message_inputs((a, b, L1) for a, b in zip(u, m))
        return rowvm.matvec_group(zip(ins, w))

    def fused():
        return rowvm.message_down(zip(u, m, w))

    a, b = current(), fused()
    for g in range(G):
        ta, tb = ttnn.to_torch(a[g]).float().reshape(-1)[:64], ttnn.to_torch(b[g]).float().reshape(-1)[:64]
        rms = lambda x: x * torch.rsqrt(x.double().square().mean() + 1e-6)
        hin = torch.cat([rms(us[g].double()).reshape(-1), rms(ms[g].double()).reshape(-1)]).bfloat16().double()
        ref = hin @ ws[g].double()
        print(
            f"G={G} gate {g}: equal {int((ta == tb).sum())}/64 err current {((ta - ref).norm() / ref.norm()):.3e} "
            f"fused {((tb - ref).norm() / ref.norm()):.3e}",
            flush=True,
        )
    for name, fn in (("current", current), ("fused", fused)):
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
        print(f"G={G} {name}: {sorted(ts)[2]:.2f} us", flush=True)
ttnn.close_device(dev)
