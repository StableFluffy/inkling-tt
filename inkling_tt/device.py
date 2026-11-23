"""Device defaults for the standalone p150a runtime."""

import os

import ttnn

WORKER_L1_SIZE = 1_400_960
L1_SMALL_SIZE = 16_384
TRACE_REGION_SIZE = 400_000_000


def open_device(device_id=0):
    return ttnn.open_device(
        device_id=device_id,
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=TRACE_REGION_SIZE,
        worker_l1_size=0 if os.getenv("INK_LARGE_DISPATCH_RING") == "0" else WORKER_L1_SIZE,
    )


def has_default_l1_budget(device):
    return ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank > WORKER_L1_SIZE
