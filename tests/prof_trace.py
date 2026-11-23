import sys

import ttnn

from inkling_tt import paths
from inkling_tt.device import open_device
from inkling_tt.model import Precision
from inkling_tt.model2 import InklingTT2

layers = int(sys.argv[1]) if len(sys.argv) > 1 else 7
dev = open_device()
prec = Precision(
    expert_dtype=ttnn.bfloat8_b,
    dense_dtype=ttnn.bfloat16,
    expert_fidelity=ttnn.MathFidelity.HiFi4,
    dense_fidelity=ttnn.MathFidelity.HiFi4,
)

prec.experts = paths.experts("stock")
m = InklingTT2(dev, paths.hf_sd(), prec, lmax=2048, layers=layers, log=lambda s: None)
m.capture()
ttnn.ReadDeviceProfiler(dev)
for p, tok in enumerate([200001, 42, 43]):
    m.step(tok, p)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
ttnn.close_device(dev)
