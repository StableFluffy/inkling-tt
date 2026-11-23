from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import torch
import ttnn

from inkling_tt.model import HD, HKV, D
from inkling_tt.model2 import BF, DRAM, FP, RM
from inkling_tt.prefill import _alloc, _grid, _program, _tasks, _tile_cb

_KERNELS = Path(__file__).resolve().parent / "kernels" / "mtp"


class StagedVisit(NamedTuple):
    k: ttnn.Tensor
    v: ttnn.Tensor
    conv: tuple[ttnn.Tensor, ...]


@lru_cache(maxsize=16)
def _commit_program(tasks, rows, accessors):
    grid, rt = _grid(tasks)
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "target_commit.cpp"),
        core_ranges=grid,
        runtime_args=rt,
        compile_time_args=[rows, *sum(accessors, ())],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [_tile_cb(grid, 0, BF, size=64 + 2048)])


class TargetCommit:
    """Commit the accepted prefix from staged BF16 KV and FP32 pre-convolution rows."""

    def __init__(self, device, visits, rows):
        if not 2 <= rows <= 8 or not visits:
            raise ValueError("Target commit requires visits and a verify capacity of 2-8 rows")
        shape = (1, HKV, 32, HD)
        self.staged = {
            slot: StagedVisit(
                _alloc(device, shape),
                _alloc(device, shape),
                tuple(_alloc(device, (1, 1, 32, width), FP) for width in (HKV * HD, HKV * HD, D, D)),
            )
            for slot in visits
        }
        self.tensors = []
        table = []
        for slot, live in visits.items():
            scratch = self.staged[slot]
            visit = (live.k, live.v, scratch.k, scratch.v, *live.conv, *scratch.conv)
            self.tensors.extend(visit)
            table.append([*[t.buffer_address() for t in visit], live.klen, 0, 0, 0])
        self.addresses = ttnn.from_torch(
            torch.tensor(table, dtype=torch.int64), dtype=ttnn.uint32, layout=RM, device=device, memory_config=DRAM
        )
        self.meta = _alloc(device, (1, 8), ttnn.uint32, RM)
        tiles_per_visit = (2 * HKV * HD + 2 * (HKV * HD + D)) // 32
        tasks = _tasks(device, len(visits) * tiles_per_visit)
        accessors = tuple(
            tuple(ttnn.TensorAccessorArgs(t).get_compile_time_args())
            for t in (self.meta, self.addresses, self.tensors[0], self.tensors[4])
        )
        self.program, kernels = _commit_program(tasks, rows, accessors)
        self.kernel = kernels[0]

    def commit(self):
        self.kernel.common_runtime_args = [self.meta.buffer_address(), self.addresses.buffer_address()]
        self.program.kernels = [self.kernel]
        ttnn.generic_op([self.meta, self.addresses, *self.tensors], self.program)
