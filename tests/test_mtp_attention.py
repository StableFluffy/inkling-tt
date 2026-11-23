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
        for length in (W, 2048):
            check(device, length)
    finally:
        ttnn.close_device(device)


def check(device, length):
    generator = torch.Generator().manual_seed(487)

    def random(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.bfloat16)

    def upload(t, mem=DRAM):
        return ttnn.from_torch(t, dtype=BF, layout=TILE, device=device, memory_config=mem)

    def copy(t, out):
        host = ttnn.from_torch(t.contiguous(), dtype=out.dtype, layout=out.layout)
        ttnn.copy_host_to_device_tensor(host, out)

    def scalar(value, out):
        copy(torch.tensor([value], dtype=torch.int32), out)

    global_cache = length != W
    q, relative, sink = (upload(random(*shape)) for shape in ((1, 1, 24, 128), (1, 1, 24, 32), (32, 32)))
    rel_table = torch.cat((random(W, 32), torch.zeros(1, 32, dtype=torch.bfloat16)))
    table = attn_decode.prepare_table(rel_table, global_cache=global_cache, device=device)
    gate = upload(random(1, 1, 1, 3072), dense._l1_shards(device, 3072, 4))
    out = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, 3072)), BF, TILE, device, gate.memory_config())
    k, v = (_alloc(device, (1, 8, length, 128)) for _ in range(2))
    staged_k, staged_v = (_alloc(device, (1, 8, 32, 128)) for _ in range(2))
    pos, start = (_alloc(device, (1,), ttnn.int32, RM) for _ in range(2))

    def run(staged=None):
        attn_decode.decode(
            q,
            k,
            v,
            relative,
            sink,
            table,
            pos,
            gate,
            global_cache=global_cache,
            gate_col=0,
            out=out,
            staged=staged,
        )

    inputs = []
    for block_start in (28, 31, 32, 1020, 1023, 1024, 1031, 2040):
        if global_cache and block_start + 7 >= length:
            continue
        old = [random(1, 8, length, 128) for _ in range(2)]
        new = [random(1, 8, 32, 128) for _ in range(2)]
        expected = []
        for r in range(8):
            for original, proposed, dest in zip(old, new, (k, v)):
                data = original.clone()
                for i in range(r + 1):
                    data[:, :, (block_start + i) % length] = proposed[:, :, i]
                copy(data, dest)
            scalar(block_start + r, pos)
            run()
            expected.append(ttnn.to_torch(out))
        for data, dest in zip([*old, *new], (k, v, staged_k, staged_v)):
            copy(data, dest)
        scalar(block_start, start)
        run((staged_k, staged_v, start))
        assert torch.equal(ttnn.to_torch(out), expected[-1]), (length, block_start)
        inputs.append((block_start, old, new, expected))

    ttnn.synchronize_device(device)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    run((staged_k, staged_v, start))
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for block_start, old, new, expected in inputs:
            for data, dest in zip([*old, *new], (k, v, staged_k, staged_v)):
                copy(data, dest)
            scalar(block_start, start)
            for r, oracle in enumerate(expected):
                scalar(block_start + r, pos)
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                assert torch.equal(ttnn.to_torch(out), oracle), (length, block_start, r)
            assert torch.equal(ttnn.to_torch(k), old[0])
            assert torch.equal(ttnn.to_torch(v), old[1])
        print(f"{length} slots: {8 * len(inputs)} traced overlays bitwise equal to serial B1 attention", flush=True)
    finally:
        ttnn.release_trace(device, trace)


if __name__ == "__main__":
    main()
