"""Warm traced prefill TTFT ladder, golden checks and an eager stage profile.

usage: bench_prefill.py <chunk> [<chunk> ...]
env: PREFILL_LMAX, PREFILL_LENGTHS=128,1024,..., PREFILL_REPS, PREFILL_GOLDEN, PREFILL_TEACHER,
     PREFILL_DECODE, PREFILL_PROFILE, PREFILL_LOGICAL_RESET, PREFILL_LAYERS,
     chunk argument "1024:experts+route+attention" replaces those stages by no-ops (timing only)
"""

import os
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps

import torch
import ttnn

import inkling_tt.prefill as pp
from inkling_tt import dense, paths
from inkling_tt.device import open_device
from inkling_tt.model import Precision
from inkling_tt.model2 import InklingTT2

SCHEDULES = {
    128: (128,),
    512: (512,),
    1024: (1024, 2048, 8192, 32768, 1024),
    2048: (2048, 8192, 32768, 2048),
}


ORIGINAL = {}


def skip(names):
    """Replace stages by no-ops for timing ablations. Outputs are wrong."""
    for owner, name in (
        (pp.experts_stream, "moe_prefill"),
        (pp.experts_stream, "route_prefill"),
        (pp, "_tiled_attention"),
    ):
        ORIGINAL.setdefault((owner, name), getattr(owner, name))
        setattr(owner, name, ORIGINAL[(owner, name)])
    if "experts" in names:
        pp.experts_stream.moe_prefill = lambda x, routing, handle, *, workspace=None: workspace.output
    if "route" in names:
        pp.experts_stream.route_prefill = lambda *args: None
    if "attention" in names:
        pp._tiled_attention = lambda *args, **kwargs: args[-1]
    if "stencil" in names:
        pp._stencil = lambda x, *args, **kwargs: x


def skip_instance(pf, names):
    from inkling_tt.model2 import f32

    if "gatemsg" in names:
        pf.gate_msg = lambda layer, s, u, m, un=None, **kwargs: f32(u)
    if "mix" in names:
        pf.mix = lambda bank, partial, qm, nq: [f32(partial)] * nq
    if "msgonly" in names:
        original = pf.gate_msg
        pf.gate_msg = lambda layer, s, u, m, un=None, **kwargs: (original(layer, s, u, m, un, **kwargs), f32(u))[1]


def main():
    dev = open_device()
    t = time.perf_counter()
    spec = Precision(
        expert_dtype=ttnn.bfloat8_b,
        dense_dtype=ttnn.bfloat16,
        expert_fidelity=ttnn.MathFidelity.HiFi4,
        dense_fidelity=ttnn.MathFidelity.HiFi4,
    )
    spec.experts = os.getenv("INK_EXPERTS", "stream")
    layers = int(os.environ["PREFILL_LAYERS"]) if os.getenv("PREFILL_LAYERS") else None
    m = InklingTT2(
        dev, paths.hf_sd(), spec, layers=layers, lmax=int(os.getenv("PREFILL_LMAX", "2048")), log=lambda _: None
    )
    print("LOAD", time.perf_counter() - t, flush=True)
    try:
        for arg in sys.argv[1:]:
            size, _, skipped = arg.partition(":")
            size = int(size)
            names = set(filter(None, skipped.split("+")))
            skip(names)
            print("CHUNK", size, "skip", skipped or "none", flush=True)
            try:
                bench(dev, m, size, names)
            except Exception:
                import traceback

                traceback.print_exc()
                if m.trace_id is not None:
                    raise
                previous = getattr(m.capture, "__self__", None)
                if isinstance(previous, pp.Prefill):
                    previous.close()
                if hasattr(m.ops, "prefill_lin"):
                    del m.ops.prefill_lin
    finally:
        ttnn.close_device(dev)


def bench(dev, m, chunk, names=()):
    t = time.perf_counter()
    pf = pp.Prefill(m, chunk)
    print("PREFILL_INIT", time.perf_counter() - t, flush=True)
    skip_instance(pf, names)
    t = time.perf_counter()
    m.capture()
    print("CAPTURE", time.perf_counter() - t, "pf_trace", pf.body_trace, flush=True)
    for name in ("DRAM", "L1", "TRACE"):
        mem = ttnn.get_memory_view(dev, getattr(ttnn.BufferType, name))
        print(
            "MEM",
            name,
            mem.num_banks,
            mem.total_bytes_per_bank,
            mem.total_bytes_allocated_per_bank,
            mem.largest_contiguous_bytes_free_per_bank,
            flush=True,
        )
    g = torch.load(paths.golden("long300.pt"), weights_only=False)
    source = g["ids"].tolist()
    logical = bool(os.getenv("PREFILL_LOGICAL_RESET"))
    if os.getenv("PREFILL_GOLDEN"):
        if not logical:
            m.reset()
        lm, pos = pf.prefill(source)
        rel, agree = [], 0
        refs = g["logits"].float()
        for i, r in enumerate(refs):
            rel.append(((lm - r).norm() / r.norm()).item())
            agree += int(lm.argmax()) == int(r.argmax())
            if i + 1 < len(refs):
                lm = m.step(g["gen"][i], pos + i)
        print(
            "GOLDEN",
            len(source),
            "rel",
            rel[0],
            "continuation",
            sum(rel[1:]) / len(rel[1:]),
            "agree",
            agree,
            len(refs),
            flush=True,
        )
    if os.getenv("PREFILL_TEACHER"):
        hits = n = 0
        for sample in torch.load(paths.golden("eval3x128.pt"), weights_only=False):
            lg, pos = pf.prefill(sample["ids"].tolist())
            prompt_hits = 0
            for i, tok in enumerate(sample["gen"]):
                same = int(lg.argmax()) == int(sample["logits"][i].argmax())
                hits += same
                prompt_hits += same
                n += 1
                if i + 1 < len(sample["gen"]):
                    lg = m.step(tok, pos + i)
            print("TEACHER_PROMPT", len(sample["ids"]), prompt_hits, len(sample["gen"]), flush=True)
        print("TEACHER", hits, n, flush=True)
    lengths = SCHEDULES.get(chunk, (chunk,))
    if os.getenv("PREFILL_LENGTHS"):
        lengths = tuple(int(v) for v in os.environ["PREFILL_LENGTHS"].split(","))
    seen = {}
    for length in lengths:
        if length > m.lmax:
            continue
        timings = []
        for _ in range(int(os.getenv("PREFILL_REPS", "3"))):
            if not logical:
                m.reset()
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            lg, pos = pf.prefill((source * ((length + len(source) - 1) // len(source)))[:length])
            tok = int(lg.argmax())
            timings.append(time.perf_counter() - t)
        median = statistics.median(timings)
        print(
            "TTFT",
            length,
            f"median={median:.6f}",
            f"tok_s={length / median:.1f}",
            f"samples={timings}",
            f"argmax={tok}",
            flush=True,
        )
        if length in seen:
            print("REUSE", length, "bitwise", torch.equal(seen[length], lg), flush=True)
        seen[length] = lg
        if os.getenv("PREFILL_DECODE") and length == 8192 and pos < m.lmax:
            lg = m.step(tok, pos)
            print("HANDOFF", pos, "finite", lg.isfinite().all().item(), "argmax", int(lg.argmax()), flush=True)
    if os.getenv("PREFILL_PROFILE"):
        profile(dev, m, pf, source)
    pf.close()
    ttnn.release_trace(dev, m.trace_id)
    m.trace_id = None
    m.logits = None
    del m.ops.prefill_lin


def profile(dev, m, pf, source):
    stage, calls = defaultdict(float), defaultdict(int)

    @contextmanager
    def measure(key):
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        yield
        ttnn.synchronize_device(dev)
        stage[key] += time.perf_counter() - t
        calls[key] += 1

    def wrap(fn, label):
        @wraps(fn)
        def timed(*args, **kwargs):
            with measure(label(args) if callable(label) else label):
                return fn(*args, **kwargs)

        return timed

    restore = []

    def patch(owner, name, label=None):
        fn = getattr(owner, name)
        restore.append((owner, name, fn))
        setattr(owner, name, wrap(fn, label or name))

    for attr in ("lin", "gate_msg", "router_logits", "mix", "attention", "moe", "hclip", "hrms", "_stage", "_readout"):
        patch(pf, attr)
    patch(m.ops, "rms", "ops.rms")
    for name in ("linear_prefill_resident", "message_prefill_down", "message_prefill_up"):
        patch(dense, name, "dense." + name)
    for name in ("route_prefill", "moe_prefill"):
        patch(pp.experts_stream, name, "experts." + name)
    patch(pp.experts_stream, "_prefill_io", lambda a: "experts.combine" if a[2] else "experts.gather")
    patch(pp.experts_stream, "_prefill_project", lambda a: "experts.gate_up" if a[0] else "experts.down")
    patch(pp, "_stencil")
    if os.getenv("PREFILL_PROFILE") == "ops":
        for name in (
            "multiply",
            "add",
            "typecast",
            "sum",
            "slice",
            "permute",
            "reshape",
            "concat",
            "pad",
            "matmul",
            "linear",
            "softmax",
            "rms_norm",
            "sigmoid",
            "copy",
            "to_memory_config",
            "generic_op",
        ):
            patch(ttnn, name, "ttnn." + name)
    launch = pp._launch

    def launch_timed(factory, *args, **kwargs):
        with measure(factory.__name__):
            return launch(factory, *args, **kwargs)

    pp._launch = launch_timed
    body, head = pf.body_trace, pf.head_trace
    pf.body_trace = pf.head_trace = None
    m.reset()
    ids = (source * ((pf.chunk + len(source) - 1) // len(source)))[: pf.chunk]
    with measure("prompt"):
        pf.prefill(ids)
    pf.body_trace, pf.head_trace = body, head
    pp._launch = launch
    for owner, name, fn in reversed(restore):
        setattr(owner, name, fn)
    for _ in range(50):
        with measure("empty_sync"):
            pass
    for label in sorted(stage):
        print("PROFILE", label, calls[label], f"{stage[label] * 1e3:.2f}", flush=True)


if __name__ == "__main__":
    main()
