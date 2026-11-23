import argparse
import json
from pathlib import Path
from time import perf_counter

import ttnn

from inkling_tt import paths
from inkling_tt.device import open_device
from inkling_tt.model import Precision
from inkling_tt.model2 import InklingTT2


def profile_decode(layers):
    dev = open_device()
    try:
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
    finally:
        ttnn.close_device(dev)


def capture_labels(capture):
    from unittest.mock import patch

    generic, begin, end = ttnn.generic_op, ttnn.begin_trace_capture, ttnn.end_trace_capture
    labels = {}
    active = None

    def begin_trace(*args, **kwargs):
        nonlocal active
        trace = begin(*args, **kwargs)
        active = labels[int(trace)] = []
        return trace

    def end_trace(*args, **kwargs):
        nonlocal active
        active = None
        return end(*args, **kwargs)

    def generic_op(tensors, program):
        if active is not None:
            source = Path(program.kernels[0].kernel_source)
            if source.parent.name == "experts" or source.name == "verify_routes.cpp":
                group = "experts"
            elif source.name in {"accept.cpp", "publish.cpp", "head_tail.cpp", "feedback.cpp"}:
                group = "decision"
            else:
                group = "other"
            active.append(group)
        return generic(tensors, program)

    with (
        patch.object(ttnn, "generic_op", generic_op),
        patch.object(ttnn, "begin_trace_capture", begin_trace),
        patch.object(ttnn, "end_trace_capture", end_trace),
    ):
        capture()
    return labels


def profile_mtp(lmax, position, reps):
    import torch

    from serve.runtime import InklingRuntime

    if position < 1 or position + 8 > lmax:
        raise ValueError("The position must leave room for two four-row target blocks")
    dev = open_device(large_dispatch_ring=lmax <= 8192)
    try:
        rt = InklingRuntime(dev, paths.hf_sd(), lmax=lmax, chunk=2048, mtp=3)
        labels = capture_labels(rt.capture)
        spec = rt.speculative
        short = spec._short(position)
        traces = {1: {"b1": int(rt.m.trace_id)}}
        for width in spec.widths:
            traces[width] = {
                "draft_core": int(spec.draft_core_traces[short]),
                "draft_scores": int(spec.draft_traces[width]),
                "verify": int(spec.verify_traces[width]),
                "context": int(spec.context_trace),
            }
        print("PROFILE_TRACES", json.dumps(traces), flush=True)
        print("PROFILE_GENERIC", json.dumps({i: labels[i] for ts in traces.values() for i in ts.values()}), flush=True)
        source = torch.load(paths.golden("capture_haiku.pt"), weights_only=False)["ids"][0].tolist()
        prompt = (source * ((position + len(source) - 1) // len(source)))[:position]
        widths = tuple(traces)
        for repeat in range(reps):
            for width in widths[repeat % len(widths) :] + widths[: repeat % len(widths)]:
                rt.reset()
                spec.start_request()
                anchor = int(rt.feed(prompt, 0, draft_context=width > 1).argmax())
                if width == 1:
                    rt.m.st.write(rt.m.embed_row(anchor), position)
                    rt.m.st.device_state = (anchor, position)
                else:
                    spec._write(anchor, position)
                    spec.pending = (anchor, position)
                ttnn.synchronize_device(dev)
                ttnn.ReadDeviceProfiler(dev)
                start = perf_counter()
                for _ in range(2):
                    anchor = rt.step_greedy(anchor, rt.pos) if width == 1 else rt.speculate(anchor, rt.pos, width)[-1]
                ttnn.synchronize_device(dev)
                elapsed = 1000 * (perf_counter() - start)
                print("PROFILE_WALL", repeat, width, elapsed, flush=True)
                ttnn.ReadDeviceProfiler(dev)
        rt.release()
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("layers", type=int, nargs="?", default=7)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--lmax", type=int, default=8192)
    parser.add_argument("--position", type=int, default=29)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    if args.mtp:
        profile_mtp(args.lmax, args.position, args.reps)
    else:
        profile_decode(args.layers)
