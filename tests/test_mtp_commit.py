from types import SimpleNamespace

import torch
import ttnn

from inkling_tt.model import HD, HKV, D, W
from inkling_tt.model2 import FP
from inkling_tt.mtp_target import TargetCommit
from inkling_tt.prefill import _alloc
from serve.runtime import DEVICE_PARAMS


def make_visit(device, capacity):
    kv_shape = (1, HKV, capacity, HD)
    return SimpleNamespace(
        klen=capacity,
        k=_alloc(device, kv_shape),
        v=_alloc(device, kv_shape),
        conv=tuple(_alloc(device, (1, 1, 4, width), FP) for width in (HKV * HD, HKV * HD, D, D)),
    )


def stage(host, target):
    source = ttnn.from_torch(host.contiguous(), layout=target.layout, dtype=target.dtype)
    ttnn.copy_host_to_device_tensor(source, target)


def stage_visits(visits, staged, generator):
    cases = []
    for slot, visit in visits.items():
        scratch = staged[slot]
        for live, proposed in zip((visit.k, visit.v, *visit.conv), (scratch.k, scratch.v, *scratch.conv)):
            dtype = torch.float32 if live.dtype == FP else torch.bfloat16
            prefix, rows = (torch.randn(tuple(t.shape), dtype=dtype, generator=generator) for t in (live, proposed))
            stage(prefix, live)
            stage(rows, proposed)
            cases.append((live, prefix, rows))
    return cases


def check_state(cases, position, rows):
    for i, (live, expected, proposed) in enumerate(cases):
        if i % 6 < 2:
            for r in range(rows):
                expected[:, :, (position + r) % expected.shape[2], :] = proposed[:, :, r, :]
        else:
            for r in range(rows):
                expected = torch.cat((proposed[:, :, r : r + 1], expected[:, :, :3]), dim=-2)
        assert torch.equal(ttnn.to_torch(live), expected), (i, position, rows)


def main():
    device = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    try:
        device.enable_program_cache()
        lmax = 2112
        visits = {0: make_visit(device, W), 5: make_visit(device, lmax)}
        commit = TargetCommit(device, visits, 8)
        generator = torch.Generator().manual_seed(546)

        def prepare(position, rows):
            stage(torch.tensor([[position, rows, 0, 0, 0, 0, 0, 0]], dtype=torch.int32), commit.meta)
            return stage_visits(visits, commit.staged, generator)

        cases = prepare(1023, 1)
        commit.commit()
        check_state(cases, 1023, 1)
        trace = ttnn.begin_trace_capture(device, cq_id=0)
        commit.commit()
        ttnn.end_trace_capture(device, trace, cq_id=0)
        try:
            for position in (31, 32, 1023, 1024, 2047, 2048, lmax - 8):
                for rows in range(1, 9):
                    cases = prepare(position, rows)
                    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                    check_state(cases, position, rows)
            print("56 traced commits matched sequential state updates across local and global tile boundaries")
        finally:
            ttnn.release_trace(device, trace)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
