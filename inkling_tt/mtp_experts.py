from functools import lru_cache
from pathlib import Path

import ttnn

from inkling_tt import experts_stream as experts
from inkling_tt.prefill import _grid, _launch, _program, _tile_cb

_KERNELS = Path(__file__).resolve().parent / "kernels" / "mtp"


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


class VerifyExperts:
    def __init__(self, device, rows, *, streaming=False):
        if not 2 <= rows <= 8:
            raise ValueError("Expert verify needs two to eight live rows")
        self.rows = rows
        self.verify_mode = 2 if streaming else 1
        self.workspace = experts.prepare_prefill(device, 32)

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
        experts._prefill_io(x, work, True)
        return work.output
