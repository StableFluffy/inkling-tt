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
        decide = RoundDecision(device, 8)
        target = tuple(_alloc(device, (1, 1, 1, 1), ttnn.uint32, RM) for _ in range(8))
        drafts = tuple(_alloc(device, (1, 1, 1, 1), ttnn.uint32, RM) for _ in range(7))
        position_input = _alloc(device, (1,), ttnn.int32, RM)
        expected_ids = list(range(42, 50))

        def prepare(position, rows):
            stage(torch.tensor([position], dtype=torch.int32), position_input)
            for i, (token, dest) in enumerate(zip(expected_ids * 2, (*target, *drafts))):
                if i == rows + 7:
                    token += 99
                stage(torch.tensor([[[[token]]]], dtype=torch.int32), dest)

        def run():
            decide.accept(position_input, target, drafts, meta)

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
                    assert decide.read() == (expected_ids[:rows], position + rows)
                    assert ttnn.to_torch(meta).flatten().tolist() == [position, rows, 0, 0, 0, 0, 0, 0]
            print("56 traced decisions matched the expected accepted ranges and emitted IDs")
        finally:
            ttnn.release_trace(device, trace)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
