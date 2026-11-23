import statistics
import time

import torch
import ttnn

from inkling_tt import attn_decode, dense
from inkling_tt.model import W
from inkling_tt.model2 import BF, DRAM, RM, TILE
from inkling_tt.prefill import _alloc
from serve.runtime import DEVICE_PARAMS


def main():
    device = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    try:
        device.enable_program_cache()
        for global_cache in (False, True):
            check(device, global_cache)
    finally:
        ttnn.close_device(device)


def check(device, global_cache):
    generator = torch.Generator().manual_seed(487)

    def random(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.bfloat16)

    def upload(t, mem=DRAM):
        return ttnn.from_torch(t, dtype=BF, layout=TILE, device=device, memory_config=mem)

    def copy(t, out):
        host = ttnn.from_torch(t.contiguous(), dtype=out.dtype, layout=out.layout)
        ttnn.copy_host_to_device_tensor(host, out)

    q, relative, sink = (upload(random(*shape)) for shape in ((1, 1, 24, 128), (1, 1, 24, 32), (32, 32)))
    rel_table = torch.cat((random(W, 32), torch.zeros(1, 32, dtype=torch.bfloat16)))
    table = attn_decode.prepare_table(rel_table, global_cache=global_cache, device=device)
    gate = upload(random(1, 1, 1, 3072), dense._l1_shards(device, 3072, 4))
    out = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, 3072)), BF, TILE, device, gate.memory_config())
    serial_len = 2208 if global_cache else W
    indexed_len = 2208 if global_cache else attn_decode.RING_SLOTS
    serial = tuple(_alloc(device, (1, 8, serial_len, 128)) for _ in range(2))
    indexed = tuple(_alloc(device, (1, 8, indexed_len, 128)) for _ in range(2))
    pos = _alloc(device, (1,), ttnn.int32, RM)

    def run(k, v):
        attn_decode.decode(q, k, v, relative, sink, table, pos, gate, global_cache=global_cache, gate_col=0, out=out)

    def write_position(position):
        copy(torch.tensor([position], dtype=torch.int32), pos)

    write_position(0)
    for t in (*indexed, *serial):
        copy(torch.zeros(tuple(t.shape), dtype=torch.bfloat16), t)
    run(*indexed)
    run(*serial)
    ttnn.synchronize_device(device)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    run(*indexed)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    serial_trace = ttnn.begin_trace_capture(device, cq_id=0)
    run(*serial)
    ttnn.end_trace_capture(device, serial_trace, cq_id=0)
    try:
        cases = (28, 31, 32, 1020, 1023, 1024, 1031, 1087, 1088, 2040, 2175, 2176, 2200)
        for start in cases:
            positions = torch.arange(max(0, start - serial_len + 1), start + 8)
            inputs = [random(1, 8, len(positions), 128) for _ in range(2)]
            originals = [torch.zeros(tuple(t.shape), dtype=torch.bfloat16) for t in serial]
            for data, dst, original in zip(inputs, indexed, originals):
                with_future = torch.zeros(tuple(dst.shape), dtype=torch.bfloat16)
                with_future[:, :, positions % indexed_len] = data
                original[:, :, positions[:-8] % serial_len] = data[:, :, :-8]
                copy(with_future, dst)
            for row in range(8):
                position = start + row
                for data, dst, original in zip(inputs, serial, originals):
                    original[:, :, position % serial_len] = data[:, :, -8 + row]
                    copy(original, dst)
                write_position(position)
                run(*serial)
                expected = ttnn.to_torch(out)
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                assert torch.equal(ttnn.to_torch(out), expected), (global_cache, start, row)
        print(f"{indexed_len} slots: {8 * len(cases)} traced rows match B1 with future writes", flush=True)
        if not global_cache:
            samples = [[], []]
            for trial in range(24):
                for which in (trial % 2, 1 - trial % 2):
                    start = time.perf_counter()
                    ttnn.execute_trace(device, (serial_trace, trace)[which], cq_id=0, blocking=True)
                    samples[which].append(1e6 * (time.perf_counter() - start))
            for label, timings in zip(("1024 slots", "1088 slots"), samples):
                timings = timings[4:]
                p95 = statistics.quantiles(timings, n=20, method="inclusive")[-1]
                print(f"{label}: {statistics.median(timings):.2f}/{p95:.2f} us median/p95", flush=True)
    finally:
        ttnn.release_trace(device, serial_trace)
        ttnn.release_trace(device, trace)


if __name__ == "__main__":
    main()
