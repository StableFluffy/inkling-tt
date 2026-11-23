from functools import lru_cache
from pathlib import Path

import ttnn

from inkling_tt.model2 import RM
from inkling_tt.prefill import _alloc, _grid, _launch, _program, _tile_cb

_KERNELS = Path(__file__).resolve().parent / "kernels" / "mtp"


@lru_cache(maxsize=8)
def _accept_program(rows, specs):
    grid, _ = _grid(((0, 0, 0, 0),))
    pos, index = specs[0][:2]
    meta, result = specs[0][-2:]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "accept.cpp"),
        core_ranges=grid,
        compile_time_args=[rows, *index, *pos, *meta, *result],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [_tile_cb(grid, 0, ttnn.uint32, size=2 * rows * 64 + 64)])


class RoundDecision:
    def __init__(self, device, rows):
        if not 2 <= rows <= 8:
            raise ValueError("A speculative round needs two to eight target rows")
        self.rows = rows
        self.result = _alloc(device, (1, 16), ttnn.uint32, RM)

    def accept(self, position, target, drafts, meta):
        """Write the commit descriptor and the accepted prefix followed by the B1 continuation."""
        if len(target) != self.rows or len(drafts) != self.rows - 1:
            raise ValueError("Target and draft decisions must match the configured round width")
        _launch(_accept_program, (self.rows,), (position, *target, *drafts, meta, self.result))

    def read(self):
        data = ttnn.to_torch(self.result).flatten().tolist()
        return data[2 : 2 + data[0]], data[1]
