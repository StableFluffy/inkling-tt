import math
import struct
from pathlib import Path

import torch
import ttnn

from inkling_tt import rowvm
from inkling_tt.model import VOCAB

_SMALL_TOP_K = 128
_STRIPE = 2048
_TILES = _STRIPE // 1024
_WORKERS = math.ceil(VOCAB / _STRIPE)
_KERNELS = Path(__file__).resolve().parent / "kernels" / "sampling"


class DecodeSampler:
    def __init__(self, logits):
        self.dev = dev = logits.device()
        self.logits = logits
        self.trace = None
        grid = dev.compute_with_storage_grid_size()
        workers = [(i % grid.x, i // grid.x) for i in range(_WORKERS)]
        cores = rowvm._rects(tuple(workers))
        self.counts = self._allocate((1, 1, _WORKERS, _STRIPE), ttnn.uint32)
        self.config = self._allocate((1, 1, 1, 8), ttnn.uint32)
        self.result = self._allocate((1, 1, 1, 8), ttnn.uint32, ttnn.L1_MEMORY_CONFIG)
        self.zeros = ttnn.from_torch(
            torch.zeros(1, 1, _WORKERS, _STRIPE, dtype=torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.tensors = (self.logits, self.counts, self.config, self.result)
        specs = [arg for t in self.tensors for arg in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        coords = [dev.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) for x, y in workers]
        rt = [(ttnn.CoreCoord(x, y), [i]) for i, (x, y) in enumerate(workers)]
        address = [t.buffer_address() for t in self.tensors]
        reader = ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "select.cpp"),
            core_ranges=cores,
            compile_time_args=[_TILES, VOCAB, _WORKERS, self.logits.shape[-1], _SMALL_TOP_K, *specs],
            runtime_args=rt,
            common_runtime_args=address + [p.x | (p.y << 16) for p in coords],
            config=ttnn.ReaderConfigDescriptor(),
        )
        compute = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
        )
        unpack = [ttnn.UnpackToDestMode.Default] * 64
        unpack[1] = unpack[16] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute.unpack_to_dest_mode = unpack
        maths = ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "math.cpp"), core_ranges=cores, compile_time_args=[_TILES], config=compute
        )
        scratch = _STRIPE * 4 + 64 * (_WORKERS + 1) + 160 + (_WORKERS + 1) * _SMALL_TOP_K * 16
        cbs = []
        for i, dt, size, count in (
            (0, ttnn.bfloat16, 2048, _TILES),
            (1, ttnn.float32, 4096, _TILES),
            (2, ttnn.uint32, 32, 1),
            (3, ttnn.uint32, 32, 1),
            (4, ttnn.uint32, scratch, 1),
            (16, ttnn.float32, 4096, _TILES),
            (17, ttnn.float32, 4096, _TILES),
        ):
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=size * count,
                    core_ranges=cores,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=size)],
                )
            )
        self.program = ttnn.ProgramDescriptor(
            kernels=[reader, maths],
            cbs=cbs,
            semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in range(2)],
        )
        self.program.custom_program_hash = ttnn.compute_program_descriptor_hash(self.program)

    def _allocate(self, shape, dtype, memory=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.ROW_MAJOR_LAYOUT, self.dev, memory)

    def stage(self, last, temp, top_k, top_p, presence, frequency, draw):
        def bits(value):
            return struct.unpack("<I", struct.pack("<f", value))[0]

        values = [
            bits(1.0 / temp if temp else 1.0),
            last,
            bits(presence),
            bits(frequency),
            min(max(top_k, 0), VOCAB) if temp else 1,
            bits(top_p if temp else 1.0),
            bits(draw),
            0,
        ]
        host = ttnn.from_torch(
            torch.tensor(values, dtype=torch.uint32).reshape(1, 1, 1, 8),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host, self.config)

    def reset(self):
        ttnn.copy_host_to_device_tensor(self.zeros, self.counts)

    def _run(self):
        ttnn.generic_op(list(self.tensors), self.program)

    def warm(self):
        raw = ttnn.from_torch(
            torch.zeros(tuple(self.logits.shape), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(raw, self.logits)
        self.reset()
        self.stage(0, 1.0, 0, 1.0, 0.0, 0.0, 0.0)
        self._run()
        ttnn.synchronize_device(self.dev)

    def capture(self):
        trace = ttnn.begin_trace_capture(self.dev, cq_id=0)
        try:
            self._run()
        except Exception:
            ttnn.end_trace_capture(self.dev, trace, cq_id=0)
            ttnn.release_trace(self.dev, trace)
            raise
        ttnn.end_trace_capture(self.dev, trace, cq_id=0)
        ttnn.synchronize_device(self.dev)
        self.trace = trace

    def release(self):
        if self.trace is not None:
            ttnn.release_trace(self.dev, self.trace)
            self.trace = None

    def sample(self):
        ttnn.execute_trace(self.dev, self.trace, cq_id=0, blocking=False)
        return int(ttnn.to_torch(self.result).reshape(-1)[0])
