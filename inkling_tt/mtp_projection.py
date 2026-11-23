from functools import lru_cache
from pathlib import Path

import ttnn

from inkling_tt.model2 import BF, DRAM, TILE
from inkling_tt.prefill import _grid, _launch, _program, _tasks, _tile_cb

_KERNELS = Path(__file__).resolve().parent / "kernels" / "mtp"


@lru_cache(maxsize=32)
def _tile_rows_program(tasks, rows, specs):
    grid, rt = _grid(tasks)
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(_KERNELS / "tile_rows.cpp"),
        core_ranges=grid,
        runtime_args=rt,
        compile_time_args=[rows, *sum(specs[0][:2], ())],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return _program([kernel], [_tile_cb(grid, 0, BF)])


class SharedProjection:
    def __init__(self, device, weight):
        self.input = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, 32, weight.k)), BF, TILE, device, weight.input_shards or DRAM
        )
        self.output = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, 32, weight.n)), BF, TILE, device, weight.output_shards or DRAM
        )
        self.input_tasks = _tasks(device, weight.k // 32)

    def project(self, weight, inputs=None):
        if inputs is not None:
            _launch(_tile_rows_program, (self.input_tasks, len(inputs)), (self.input, *inputs))
        return ttnn.linear(
            self.input,
            weight.tensor,
            program_config=weight.program,
            compute_kernel_config=weight.compute,
            memory_config=self.output.memory_config(),
            optional_output_tensor=self.output,
        )
