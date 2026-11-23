from functools import lru_cache
from pathlib import Path

import torch
import ttnn

from inkling_tt import experts_stream as experts
from inkling_tt.model import D
from inkling_tt.model2 import BF, TILE
from inkling_tt.prefill import _compute, _grid, _launch, _program, _tasks, _tile_cb

_KERNELS = Path(__file__).resolve().parent / "kernels" / "mtp"
_EXPERT_KERNELS = _KERNELS.parent / "experts"


@lru_cache(maxsize=16)
def _pack_program(rows, specs):
    grid, _ = _grid(((0, 0, 0, 0),))
    ids, gamma = specs[0][:2]
    out_ids, out_gamma = specs[0][-2:]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "pack_routes.cpp"),
        core_ranges=grid,
        compile_time_args=[rows, *ids, *gamma, *out_ids, *out_gamma],
        config=ttnn.ReaderConfigDescriptor(),
    )
    size = 64 * rows + 256 * 4 + 8 * 4096
    return _program([kernel], [_tile_cb(grid, 0, ttnn.uint32, size=size)])


@lru_cache(maxsize=16)
def _combine_program(tasks, specs):
    grid, rt = _grid(tasks)
    io = str(_EXPERT_KERNELS / "verify_combine.cpp")
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=io,
            core_ranges=grid,
            compile_time_args=list(sum(specs[0], ())),
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_EXPERT_KERNELS / "compute.cpp"),
            core_ranges=grid,
            compile_time_args=[1, 1, 1],
            runtime_args=rt,
            defines=[("EXPERTS_GATE", "0"), ("EXPERTS_VERIFY_COMBINE", "1")],
            config=_compute(()),
        ),
        ttnn.KernelDescriptor(
            kernel_source=io,
            core_ranges=grid,
            compile_time_args=list(sum(specs[2][:2], ())),
            runtime_args=rt,
            defines=[("EXPERTS_VERIFY_WRITE", "1")],
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    buffers = [
        _tile_cb(grid, 0, BF, 4),
        _tile_cb(grid, 2, ttnn.uint32, size=1024),
        _tile_cb(grid, 3, ttnn.uint32, size=64),
        _tile_cb(grid, 16, BF, 2),
    ]
    return _program(kernels, buffers)


class VerifyExperts:
    def __init__(self, device, rows, *, streaming=False):
        if not 2 <= rows <= 8:
            raise ValueError("Expert verify needs two to eight live rows")
        self.rows = rows
        self.verify_mode = 2 if streaming else 1
        self.workspace = experts.prepare_prefill(device, 32)
        zeros = ttnn.from_torch(torch.zeros(1, 1, 32, D), dtype=BF, layout=TILE)
        ttnn.copy_host_to_device_tensor(zeros, self.workspace.output)

    def run(self, x, routing, handle):
        """Stream the union of B1 routes while preserving each token's combine order."""
        if len(routing) != self.rows:
            raise ValueError("Expected one set of B1 IDs and weights per verify row")
        work = self.workspace
        route_tensors = tuple(t for pair in routing for t in pair)
        _launch(_pack_program, (self.rows,), (*route_tensors, work.ids, work.gamma))
        maps = (work.ids, work.ends, work.gather, work.inverse)
        with experts._LOCK:
            desc, kernel = experts._prefill_route_program(
                False,
                ((0, 0, 0, 1),),
                tuple(experts._accessor(t) for t in maps),
                1,
                work.gather.shape[-1] * 4,
                work.rows,
            )
            kernel.common_runtime_args = [t.buffer_address() for t in maps]
            desc.kernels = [kernel]
            ttnn.generic_op(list(maps), desc)
        experts._prefill_io(x, work, False)
        experts._prefill_project(True, work, handle, verify=self.verify_mode)
        experts._prefill_project(False, work, handle, verify=self.verify_mode)
        _launch(
            _combine_program,
            (_tasks(x.device(), self.rows * (D // 32)),),
            (work.inverse, work.wide),
            (),
            (work.output, *(gamma for _, gamma in routing)),
        )
        return work.output
