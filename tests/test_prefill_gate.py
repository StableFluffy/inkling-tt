"""Compare the fused message-gate epilogue with the TTNN reference sequence."""

from statistics import median
from time import perf_counter

import torch
import ttnn

import inkling_tt.prefill as pp
from inkling_tt import dense
from inkling_tt.model2 import FP, TILE, act


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


def main():
    torch.manual_seed(5)
    dev = ttnn.open_device(device_id=0, l1_small_size=16384)
    dev.enable_program_cache()
    up = lambda x: ttnn.from_torch(x.contiguous(), dtype=FP, layout=TILE, device=dev)
    try:
        for t in (128, 1024):
            h = up(torch.randn(1, 1, t, 64).bfloat16().float())
            wf, wg = up(torch.randn(1, 1, 64, 2304) * 0.2), up(torch.randn(1, 1, 64, 2304) * 0.2)
            u, m = up(torch.randn(1, 1, t, 2304)), up(torch.randn(1, 1, t, 2304))

            def reference():
                forget = dense.message_prefill_up(h, wf)
                message = dense.message_prefill_up(h, wg)
                kept = ttnn.multiply(u, forget, input_tensor_b_activations=[act("TANH"), act("RSUB", 1.0)])
                return ttnn.add(kept, ttnn.multiply(m, message, input_tensor_b_activations=[act("TANH")]))

            a = ttnn.to_torch(reference()).float()
            b = ttnn.to_torch(pp._message_gate(h, wf, wg, u, m)).float()
            print(
                "GATE",
                t,
                "equal_frac",
                (a == b).float().mean().item(),
                "max_abs",
                (a - b).abs().max().item(),
                "rel",
                ((a - b).norm() / a.norm()).item(),
                "ms_ref",
                round(bench(dev, reference), 3),
                "ms_fused",
                round(bench(dev, lambda: pp._message_gate(h, wf, wg, u, m)), 3),
                flush=True,
            )
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
