"""Bitwise check: rowvm with row-major L1 rows versus the TILE layout, for the fused decode mixes."""

import math
import sys

import torch
import ttnn

from inkling_tt import dense, rowvm

FP, BF, TILE, RM = ttnn.float32, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
D = 2304
dev = ttnn.open_device(device_id=0, l1_small_size=16384)
torch.manual_seed(0)


def tile(t, dtype=FP, mem=DRAM):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=TILE, device=dev, memory_config=mem)


def rm(t, dtype=FP):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=RM,
        device=dev,
        memory_config=DRAM if __import__("os").environ.get("RM_DRAM") == "1" else L1,
    )


def host(t):
    return ttnn.to_torch(t).float()


bank_t = torch.randn(1, 1, 32, D)
bank_t = bank_t * torch.rsqrt(bank_t.square().mean(-1, keepdim=True) + 1e-6)
data = dict(
    bank=bank_t,
    q=torch.randn(1, 1, 1, D) / math.sqrt(D),
    q4=torch.randn(1, 1, 4, D) / math.sqrt(D),
    w=torch.randn(1, 1, 1, D),
    taps=torch.randn(1, 1, 4, D) / 3,
    res=torch.randn(1, 1, 1, D),
    entry=torch.randn(1, 1, 1, D),
)
hist0 = torch.randn(1, 1, 4, D)
partial = torch.randn(1, 1, 1, D)
shards = dense._l1_shards(dev, D, 2)
fails = 0


def run(mode, J, start, append):
    mk = tile if mode == "tile" else rm
    t = {k: mk(v) for k, v in data.items()}
    bank = t["bank"]
    a = ttnn.to_memory_config(tile(partial, BF), shards)
    hist = tile(hist0)
    z = mk(torch.zeros(1, 1, 1, D), BF if append else FP)
    xm = tile(torch.zeros(1, 1, 1, D), BF, L1)
    entry = mk(data["entry"], BF) if mode != "tile" else tile(data["entry"], BF)
    rowvm.attnres_mix(
        bank,
        a,
        t["q"],
        J,
        weight=t["w"],
        dtype=BF,
        out=(xm,),
        memory_config=L1,
        convolution=(hist, t["taps"], None if append else t["res"], z),
        bank_start=start,
        shared_prefix=bool(start),
        append=(entry, start + J - 1) if append else None,
    )
    hist2 = tile(hist0)
    y = tile(torch.zeros(1, 1, 1, D), BF)
    h = rowvm.attnres_mix(
        bank,
        tile(torch.randn(1, 1, 1, D, generator=torch.Generator().manual_seed(3)), BF, L1),
        t["q"],
        J + int(append),
        bank_start=start,
        shared_prefix=bool(start),
        weight=t["w"],
        dtype=BF,
        convolution=(hist2, t["taps"], z, y),
        round_residual=True,
        memory_config=DRAM,
    )[0]
    msgs = rowvm.attnres_mix(bank, y, t["q4"], J + int(append), bank_start=start + 1 if start else 0)
    return [host(x) for x in (xm, z, hist, bank, hist2, y, h, *msgs)]


for J, start, append in ((1, 0, False), (5, 3, False), (12, 1, False), (4, 1, True), (7, 5, True)):
    ref, got = run("tile", J, start, append), run("rm", J, start, append)
    same = [bool(torch.equal(a, b)) for a, b in zip(ref, got)]
    print(f"J={J} start={start} append={append} bitwise={same}", flush=True)
    fails += not all(same)
ttnn.close_device(dev)
sys.exit(1 if fails else 0)
