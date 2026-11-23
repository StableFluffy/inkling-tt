import statistics
import time

import torch
import ttnn

from inkling_tt import dense, paths, rowvm
from inkling_tt.model import VOCAB, D, Weights
from inkling_tt.model2 import BF, RM, TILE
from inkling_tt.mtp_weights import PADDED_VOCAB
from inkling_tt.prefill import _alloc
from serve.runtime import DEVICE_PARAMS


def main():
    device = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    try:
        device.enable_program_cache()
        weight = Weights(paths.hf_sd()).get("lm_head.weight")
        head = dense.prepare_weight(device, torch.nn.functional.pad(weight.T, (0, PADDED_VOCAB - VOCAB)))
        generator = torch.Generator().manual_seed(869)
        cases = [torch.randn(1, 1, 32, D, generator=generator, dtype=torch.bfloat16) for _ in range(2)]

        def workspace(rows):
            x_shape = (1, 1, rows, D)
            out_shape = (1, 1, rows, PADDED_VOCAB)
            return (
                _alloc(device, x_shape),
                ttnn.allocate_tensor_on_device(ttnn.Shape(x_shape), BF, TILE, device, head.input_shards),
                ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), BF, TILE, device, head.output_shards),
            )

        shared = workspace(32)
        serial = [workspace(1) for _ in range(4)]
        outputs = [
            [
                (_alloc(device, (1, 1, 1, PADDED_VOCAB), BF, RM), _alloc(device, (1, 1, 1, 1), ttnn.uint32, RM))
                for _ in range(4)
            ]
            for _ in range(2)
        ]

        def stage(src, dst):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(src.contiguous(), dtype=BF, layout=TILE), dst)

        def prepare(block):
            stage(block, shared[0])
            for i, (x, _, _) in enumerate(serial):
                stage(block[:, :, i : i + 1], x)

        def project(tensors):
            x, shards, result = tensors
            ttnn.interleaved_to_sharded(x, head.input_shards, preallocated_output=shards)
            ttnn.linear(
                shards,
                head.tensor,
                program_config=head.program,
                compute_kernel_config=head.compute,
                memory_config=head.output_shards,
                optional_output_tensor=result,
            )
            return result

        def narrow():
            for tensors, (out, index) in zip(serial, outputs[0]):
                rowvm.head_tail(project(tensors), out, index, VOCAB)

        def wide():
            result = project(shared)
            for i, (out, index) in enumerate(outputs[1]):
                rowvm.head_tail(result, out, index, VOCAB, row=i)

        def check():
            for lhs, rhs in zip(*outputs):
                for a, b in zip(lhs, rhs):
                    assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b))

        prepare(cases[0])
        narrow()
        wide()
        check()
        ttnn.synchronize_device(device)
        traces = []
        for graph in (narrow, wide):
            trace = ttnn.begin_trace_capture(device, cq_id=0)
            graph()
            ttnn.end_trace_capture(device, trace, cq_id=0)
            traces.append(trace)
        try:
            timings = [[], []]
            for i in range(36):
                prepare(cases[i % 2])
                ttnn.synchronize_device(device)
                for k in (0, 1) if i % 2 == 0 else (1, 0):
                    start = time.perf_counter()
                    ttnn.execute_trace(device, traces[k], cq_id=0, blocking=True)
                    timings[k].append(1000 * (time.perf_counter() - start))
                check()
            for name, samples in zip(("serial", "shared"), timings):
                print(
                    f"{name}: {statistics.median(samples[4:]):.3f} ms median, "
                    f"{statistics.quantiles(samples[4:], n=20)[-1]:.3f} ms p95",
                    flush=True,
                )
        finally:
            for trace in traces:
                ttnn.release_trace(device, trace)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
