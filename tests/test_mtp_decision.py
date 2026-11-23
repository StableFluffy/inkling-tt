import torch
import ttnn

from inkling_tt.model2 import RM
from inkling_tt.mtp_round import RoundDecision
from inkling_tt.prefill import _alloc
from serve.runtime import DEVICE_PARAMS


def stage(host, target):
    source = ttnn.from_torch(host.contiguous(), layout=target.layout, dtype=target.dtype)
    ttnn.copy_host_to_device_tensor(source, target)


def main():
    device = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    try:
        device.enable_program_cache()
        lmax = 2112
        meta = _alloc(device, (1, 8), ttnn.uint32, RM)
        decide = RoundDecision(device, 8, adaptive=True)
        target = tuple(_alloc(device, (1, 1, 1, 1), ttnn.uint32, RM) for _ in range(8))
        drafts = tuple(_alloc(device, (1, 1, 1, 1), ttnn.uint32, RM) for _ in range(7))
        position_input = _alloc(device, (1,), ttnn.int32, RM)
        anchor = _alloc(device, (1, 1), ttnn.uint32, RM)
        expected_ids = list(range(42, 50))

        def prepare(position, rows):
            stage(torch.tensor([position], dtype=torch.int32), position_input)
            decide.set_policy([lmax, 4, 1024, 100, 158, 194, 240, 100, 140, 175, 210, 6963, 1229, 0, 0, 0])
            for i, (token, dest) in enumerate(zip(expected_ids * 2, (*target, *drafts))):
                if i == rows + 7:
                    token += 99
                stage(torch.tensor([[[[token]]]], dtype=torch.int32), dest)

        def run():
            decide.accept(position_input, target, drafts, meta)
            decide.publish(anchor, position_input)

        for count in (1, 2):
            prepare(29, count)
            decide.set_policy(
                [
                    lmax,
                    4,
                    1024,
                    20_000,
                    35_000,
                    42_000,
                    50_000,
                    20_000,
                    35_000,
                    42_000,
                    50_000,
                    6963,
                    1229,
                    0,
                    15_000,
                    0,
                ]
            )
            decide.accept(position_input, target[:2], drafts[:1], meta)
            decide.publish(anchor, position_input)
            assert decide.read(include_width=True) == (expected_ids[:count], 29 + count, count)
        print("A rejected narrow probe exhausts the replay budget", flush=True)

        prepare(1023, 1)
        run()
        assert decide.read() == (expected_ids[:1], 1024)
        trace = ttnn.begin_trace_capture(device, cq_id=0)
        run()
        ttnn.end_trace_capture(device, trace, cq_id=0)
        try:
            for position in (31, 32, 1023, 1024, 2047, 2048, lmax - 8):
                for rows in range(1, 9):
                    prepare(position, rows)
                    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                    end = position + rows
                    costs = [100, 140, 175, 210] if end <= 1024 else [100, 158, 194, 240]
                    accept = (6963 * 9 + 5) // 10 + (rows - 1) * 4096
                    reject = (1229 * 9 + 5) // 10 + int(rows < 8) * 4096
                    acceptance = accept * 4096 // (accept + reject)
                    power, expected = 4096, [4096]
                    for _ in range(3):
                        power = power * acceptance // 4096
                        expected.append(expected[-1] + power)
                    choices = range(min(4, max(1, lmax - end)))
                    next_width = 1 + max(choices, key=lambda width: expected[width] / costs[width])
                    assert decide.read(include_width=True) == (expected_ids[:rows], end, next_width)
                    assert ttnn.to_torch(anchor).item() == expected_ids[rows - 1]
                    assert ttnn.to_torch(position_input).item() == end
                    assert ttnn.to_torch(meta).flatten().tolist() == [position, rows, 0, 0, 0, 0, 0, 0]
            print("56 traced decisions matched the expected IDs, next position and adaptive width")
        finally:
            ttnn.release_trace(device, trace)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
