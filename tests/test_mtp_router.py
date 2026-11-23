import statistics
import time

import torch
import ttnn

from inkling_tt import paths, rowvm
from inkling_tt.model import D, Weights
from inkling_tt.model2 import BF, DRAM, FP, L1, RM, TILE
from inkling_tt.mtp_projection import _tile_rows_program
from inkling_tt.prefill import _alloc, _launch, _tasks
from serve.runtime import DEVICE_PARAMS


def main():
    device = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    try:
        device.enable_program_cache()
        weights = Weights(paths.hf_sd())
        weight = ttnn.from_torch(
            weights.get("layers.0.mlp.gate.weight").T.contiguous(), dtype=BF, layout=TILE, device=device
        )
        bias = ttnn.from_torch(
            weights.get("layers.0.mlp.gate.e_score_correction_bias").reshape(1, 1, 1, 128),
            dtype=FP,
            layout=RM,
            device=device,
            memory_config=DRAM,
        )
        x = torch.randn(4, D, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(91))
        live = tuple(
            ttnn.from_torch(t.reshape(1, 1, 1, D), dtype=BF, layout=TILE, device=device, memory_config=L1) for t in x
        )
        packed = _alloc(device, (1, 1, 32, D))
        routes = tuple(
            (
                ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, 8)), ttnn.uint16, RM, device, L1),
                ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, 8)), FP, TILE, device, L1),
            )
            for _ in live
        )
        tasks = _tasks(device, D // 32)

        def pack():
            _launch(_tile_rows_program, (tasks, len(live)), (packed, *live))

        def serial():
            pack()
            for row, (ids, gamma) in zip(live, routes):
                rowvm.fused_router(row, weight, bias, ids_out=ids, gamma_out=gamma)

        def shared():
            pack()
            rowvm.fused_router_block(packed, weight, bias, routes)

        def read():
            return [ttnn.to_torch(t) for route in routes for t in route]

        serial()
        expected = read()
        shared()
        for k, (a, b) in enumerate(zip(read(), expected)):
            assert torch.equal(a, b), (k, a, b)
        print("all four router ID and FP32 gamma rows match B1 bitwise", flush=True)
        traces = []
        for body in (serial, shared):
            trace = ttnn.begin_trace_capture(device, cq_id=0)
            body()
            ttnn.end_trace_capture(device, trace, cq_id=0)
            traces.append(trace)
        timings = [[], []]
        try:
            for r in range(40):
                for k in (0, 1) if r % 2 == 0 else (1, 0):
                    start = time.perf_counter()
                    ttnn.execute_trace(device, traces[k], cq_id=0, blocking=True)
                    timings[k].append(1e6 * (time.perf_counter() - start))
                    assert all(torch.equal(a, b) for a, b in zip(read(), expected))
            for name, t in zip(("serial", "shared"), timings):
                p95 = statistics.quantiles(t[4:], n=20, method="inclusive")[-1]
                print(f"{name} router: {statistics.median(t[4:]):.2f}/{p95:.2f} us median/p95", flush=True)
        finally:
            for trace in traces:
                ttnn.release_trace(device, trace)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
