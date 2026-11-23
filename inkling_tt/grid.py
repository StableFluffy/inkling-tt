"""Worker-grid setup for Inkling's Blackhole programs."""

import os
import re

MIN_COLS = 11
MIN_ROWS = 10
_METAL_OVERRIDE = "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"


def _check_minimum(cols, rows):
    if cols < MIN_COLS or rows < MIN_ROWS:
        raise ValueError(f"Inkling requires at least an {MIN_COLS}x{MIN_ROWS} worker grid, got {cols}x{rows}")


def configure_grid():
    """Apply INK_GRID before TT-Metal first initializes its device runtime."""
    override = os.environ.get("INK_GRID")
    if override is None:
        return None
    match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", override)
    if match is None:
        raise ValueError(f"INK_GRID must be <cols>x<rows>, got {override!r}")
    cols, rows = map(int, match.groups())
    _check_minimum(cols, rows)
    # This descriptor setting also bounds TTNN factories that choose their own
    # cores, including the DRAM-sharded matmul pipeline.
    end = f"{cols - 1},{rows - 1}"
    existing = os.environ.setdefault(_METAL_OVERRIDE, end)
    if existing != end:
        raise ValueError(f"INK_GRID={override} conflicts with {_METAL_OVERRIDE}={existing}")
    return cols, rows


_requested = configure_grid()


def worker_grid(device):
    grid = device.compute_with_storage_grid_size()
    _check_minimum(grid.x, grid.y)
    if _requested is not None and (grid.x, grid.y) != _requested:
        raise ValueError(
            f"INK_GRID={_requested[0]}x{_requested[1]} requires initializing TT-Metal after importing "
            f"inkling_tt.grid in a fresh process. Device reports {grid.x}x{grid.y}."
        )
    return grid
