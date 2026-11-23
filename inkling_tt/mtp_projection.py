from functools import lru_cache
from pathlib import Path

import ttnn

from inkling_tt.model2 import BF, DRAM, FP, TILE
from inkling_tt.prefill import _grid, _launch, _program, _tasks, _tile_cb

_KERNELS = Path(__file__).resolve().parent / "kernels" / "mtp"


@lru_cache(maxsize=64)
def _tile_row_program(tasks, dtype, src_row, dst_row, specs):
    grid, rt = _grid(tasks)
    tile = 4096 if dtype == FP else 2048
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "tile_row.cpp"),
        core_ranges=grid,
        runtime_args=rt,
        compile_time_args=[tile, src_row, dst_row, *sum(specs[0], ())],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [_tile_cb(grid, 0, dtype)])


def _tile_row(src, dst, *, source=0, target=0):
    args = (_tasks(src.device(), src.shape[-1] // 32), src.dtype, source, target)
    _launch(_tile_row_program, args, (src, dst))


def _tile_rows_kernel(tasks, rows, specs, group=None):
    grid, rt = _grid(tasks)
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "tile_rows.cpp"),
        core_ranges=grid,
        runtime_args=rt,
        compile_time_args=[rows, *sum(specs[:2], ())],
        defines=[] if group is None else [("MTP_PACK_GROUP", str(group))],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return kernel, _tile_cb(grid, 0, BF)


@lru_cache(maxsize=32)
def _tile_rows_program(tasks, rows, specs):
    kernel, cb = _tile_rows_kernel(tasks, rows, specs[0])
    return _program([kernel], [cb])


@lru_cache(maxsize=16)
def _tile_rows_group_program(tasks, rows, specs):
    kernels, buffers = zip(
        *(_tile_rows_kernel(group, rows, spec, i) for i, (group, spec) in enumerate(zip(tasks, specs)))
    )
    return _program(list(kernels), list(buffers))


def _projection_tasks(device, cols, groups):
    grid = device.compute_with_storage_grid_size()
    workers = min(grid.x * grid.y // groups, cols)
    return tuple(
        tuple(
            ((base + i) % grid.x, (base + i) // grid.x, i * cols // workers, (i + 1) * cols // workers)
            for i in range(workers)
        )
        for base in range(0, groups * workers, workers)
    )


def _pack_projections(tasks, inputs):
    _launch(_tile_rows_group_program, (tasks, len(inputs[0]) - 1), *inputs)


def _pack_rows(matrix, inputs, tasks):
    _launch(_tile_rows_program, (tasks, len(inputs)), (matrix, *inputs))


class SharedProjection:
    def __init__(self, device, weight):
        self.input = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, 32, weight.k)), BF, TILE, device, weight.input_shards or DRAM
        )
        self.output = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, 32, weight.n)), BF, TILE, device, weight.output_shards or DRAM
        )
        self.input_tasks = _tasks(device, weight.k // 32)

    def project(self, weight, inputs=None, *, activation=None):
        if inputs is not None:
            _pack_rows(self.input, inputs, self.input_tasks)
        return ttnn.linear(
            self.input if activation is None else activation,
            weight.tensor,
            program_config=weight.program,
            compute_kernel_config=weight.compute,
            memory_config=self.output.memory_config(),
            optional_output_tensor=self.output,
        )
