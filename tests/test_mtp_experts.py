import statistics
import time

import torch
import ttnn

from inkling_tt import experts_stream, paths
from inkling_tt.model import D, Weights
from inkling_tt.model2 import BF, FP, L1, RM, TILE
from inkling_tt.mtp_experts import VerifyExperts
from inkling_tt.prefill import _alloc
from serve.runtime import DEVICE_PARAMS


def stage(data, dest):
    host = ttnn.from_torch(data.contiguous(), dtype=dest.dtype, layout=dest.layout)
    ttnn.copy_host_to_device_tensor(host, dest)


def main():
    rows = 4
    generators = [torch.Generator().manual_seed(i) for i in (456, 457, 458)]
    device = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    try:
        device.enable_program_cache()
        weights = Weights(paths.hf_sd())
        handle = experts_stream.prepare(
            device, weights.get("layers.0.mlp.experts.gate_up_proj"), weights.get("layers.0.mlp.experts.down_proj")
        )
        verify = VerifyExperts(device, rows)
        packed = _alloc(device, (1, 1, 32, D))
        routing = [
            (
                ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, 8)), ttnn.uint16, RM, device, L1),
                ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, 8)), FP, TILE, device, L1),
            )
            for _ in range(rows)
        ]

        def prepare(x, ids, gamma):
            stage(x, packed)
            for i, (dev_ids, dev_gamma) in enumerate(routing):
                stage(ids[i].reshape(1, 1, 1, 8), dev_ids)
                stage(gamma[i].reshape(1, 1, 1, 8), dev_gamma)

        cases = []
        for stride, generator in zip((0, 4, 8), generators):
            x = torch.randn(1, 1, 32, D, dtype=torch.bfloat16, generator=generator)
            ids = torch.stack([stride * i + torch.randperm(8, generator=generator) for i in range(rows)])
            gamma = torch.rand(rows, 8, generator=generator)
            gamma *= 8 / gamma.sum(-1, keepdim=True)
            prepare(x, ids, gamma)
            output = verify.run(packed, routing, handle)
            data = ttnn.to_torch(output)[0, 0]
            refs = []
            for i, (dev_ids, dev_gamma) in enumerate(routing):
                row = ttnn.from_torch(x[:, :, i : i + 1], dtype=BF, layout=TILE, device=device, memory_config=L1)
                refs.append(ttnn.to_torch(experts_stream.moe_decode(row, dev_ids, dev_gamma, handle))[0, 0, 0])
            expected = torch.stack(refs)
            union = ids.unique().numel()
            failures = (data[:rows] != expected).sum().item()
            print(f"union {union}: {failures}/{rows * D} mismatches against B1", flush=True)
            assert torch.equal(data[:rows], expected)
            assert torch.count_nonzero(data[rows:]) == 0
            cases.append((x, ids, gamma, expected))

        ttnn.synchronize_device(device)
        trace = ttnn.begin_trace_capture(device, cq_id=0)
        verify.run(packed, routing, handle)
        ttnn.end_trace_capture(device, trace, cq_id=0)
        try:
            samples = [[] for _ in cases]
            for i in range(18):
                j = i % len(cases)
                x, ids, gamma, expected = cases[j]
                prepare(x, ids, gamma)
                ttnn.synchronize_device(device)
                start = time.perf_counter()
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                samples[j].append(1000 * (time.perf_counter() - start))
                assert torch.equal(ttnn.to_torch(verify.workspace.output)[0, 0, :rows], expected)
            for (_, ids, _, _), times in zip(cases, samples):
                print(
                    f"union {ids.unique().numel()}: {statistics.median(times[1:]):.3f} ms median, "
                    f"{max(times[1:]):.3f} ms p95",
                    flush=True,
                )
        finally:
            ttnn.release_trace(device, trace)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
