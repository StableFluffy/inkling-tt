"""Time the packed prefill expert phases on seeded activations with real routers.

usage: bench_experts_prefill.py [tokens ...]   env EXPERT_VARIANTS=name:LANES=7+WIDE=320,...
"""

import os
import sys
from statistics import median
from time import perf_counter

import torch
import ttnn
from safetensors import safe_open

from inkling_tt import experts_stream as es


def bench(dev, fn, reps=7):
    fn()
    ttnn.synchronize_device(dev)
    times = []
    for _ in range(reps):
        start = perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        times.append((perf_counter() - start) * 1000)
    return round(median(times), 3)


def variants():
    spec = os.getenv("EXPERT_VARIANTS", "base:")
    out = []
    for item in spec.split(","):
        name, _, defs = item.partition(":")
        out.append((name, tuple(tuple(d.split("=")) for d in defs.split("+") if d)))
    return out


def main():
    torch.manual_seed(42)
    dev = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=20_000_000)
    dev.enable_program_cache()
    sizes = [int(v) for v in sys.argv[1:]] or [1024, 2048]
    try:
        for i in (0, 20):
            prefix = f"layers.{i}.mlp."
            with safe_open(f"hf_sd/layer{i:02}.safetensors", framework="pt") as s:
                handle = es.prepare(
                    dev, s.get_tensor(prefix + "experts.gate_up_proj"), s.get_tensor(prefix + "experts.down_proj")
                )
                rw = s.get_tensor(prefix + "gate.weight").float()
                rb = s.get_tensor(prefix + "gate.e_score_correction_bias").float()
            for t in sizes:
                v = torch.randn(1, 1, t, 2304).bfloat16()
                x = ttnn.from_torch(v, device=dev, layout=ttnn.TILE_LAYOUT)
                scores = torch.sigmoid(v.reshape(t, 2304).float() @ rw.T)
                ids = torch.topk(scores + rb, 8).indices
                gamma = scores.gather(1, ids)
                gamma = gamma * 8 / gamma.sum(-1, keepdim=True)
                expected = None
                for name, defs in variants():
                    opts = dict(defs)
                    lanes = int(opts.pop("LANES", 0))
                    wide = opts.pop("WIDE", None)
                    ws = es.prepare_prefill(
                        dev,
                        t,
                        broadcast_down=t >= 512,
                        full_tiles=t >= 512,
                        lanes=lanes,
                        wide_l1_blocks=None if wide is None else int(wide),
                    )
                    es.stage_prefill_routing((ids, gamma), ws)
                    y = ttnn.to_torch(es.moe_prefill(x, None, handle, workspace=ws))
                    if expected is None:
                        expected = y
                    same = torch.equal(y, expected)
                    full = bench(dev, lambda: es.moe_prefill(x, None, handle, workspace=ws))
                    gather = bench(dev, lambda: es._prefill_io(x, ws, False))
                    project = es._prefill_project_lanes if lanes else es._prefill_project
                    gate = bench(dev, lambda: project(True, ws, handle))
                    down = bench(dev, lambda: project(False, ws, handle))
                    combine = bench(dev, lambda: es._prefill_io(x, ws, True))
                    print(
                        "EXPERTS",
                        i,
                        t,
                        name,
                        "bitwise",
                        same,
                        "full",
                        full,
                        "gather",
                        gather,
                        "gate",
                        gate,
                        "down",
                        down,
                        "combine",
                        combine,
                        flush=True,
                    )
                    ws = None
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
