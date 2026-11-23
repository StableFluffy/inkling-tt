"""Capture warmed rowvm and model2-equivalent stock chains, then replay each."""

import argparse
import json
import math
import statistics
import time

import torch
import ttnn
from test_rowvm import BF, DRAM, FP, L1, rms, upload

from inkling_tt import rowvm

D = 2304
EPS = 1e-6


def cases(dev):
    ck = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def rnd(shape, dtype=FP, mem=DRAM):
        return upload(dev, torch.randn(shape), dtype, mem)

    def row(width=D, dtype=FP):
        return rnd((1, 1, 1, width), dtype)

    def f32(t):
        return t if t.dtype == FP else ttnn.typecast(t, FP)

    def bf16(t):
        return t if t.dtype == BF else ttnn.typecast(t, BF)

    def act(name, value=None):
        op = getattr(ttnn.UnaryOpType, name)
        return ttnn.UnaryWithParam(op) if value is None else ttnn.UnaryWithParam(op, value)

    def stock_rms(t):
        return ttnn.rms_norm(f32(t), epsilon=EPS, compute_kernel_config=ck)

    def mm(a, b, **kw):
        return ttnn.matmul(a, b, compute_kernel_config=ck, dtype=FP, **kw)

    x, w = row(dtype=BF), row()
    yield "rms2304", lambda: rowvm.rms_norm(x, w, dtype=BF), lambda: ttnn.multiply(stock_rms(x), w, dtype=BF)
    x, w = row(4608), row(4608)
    yield "rms4608", lambda: rowvm.rms_norm(x, w, dtype=BF), lambda: ttnn.multiply(stock_rms(x), w, dtype=BF)

    for nq, j in ((1, 4), (3, 4), (4, 7)):
        p = row(dtype=BF)
        bank = rms(torch.randn(1, 1, 32, D))
        bank[:, :, j:] = 0
        s = upload(dev, bank)
        q = torch.randn(1, 1, nq, D) / math.sqrt(D)
        q_pad = torch.zeros(1, 1, D, 32)
        q_pad[:, :, :, :nq] = q.mT
        q_rows, q_col = upload(dev, q), upload(dev, q_pad)
        sel = torch.zeros(1, 1, 32, 1)
        sel[:, :, j, :] = 1
        mask = torch.zeros_like(sel)
        mask[:, :, j + 1 :, :] = -1e9
        selector, invalid = upload(dev, sel), upload(dev, mask)

        def stock_mix():
            samples = ttnn.add(s, mm(selector, stock_rms(p)))
            dot = ttnn.sum(ttnn.multiply(samples, q_rows), dim=-1, keepdim=True) if nq == 1 else mm(samples, q_col)
            prob = ttnn.softmax(ttnn.add(dot, invalid), dim=-2, numeric_stable=True, compute_kernel_config=ck)
            all_out = mm(prob, samples, transpose_a=True)
            return tuple(ttnn.slice(all_out, [0, 0, i, 0], [1, 1, i + 1, D]) for i in range(nq))

        yield f"mix{nq}_j{j}", lambda: rowvm.attnres_mix(s, p, q_rows, j), stock_mix
        if nq == 3:
            ww = torch.randn(1, 1, nq, D)
            weights = upload(dev, ww)
            individual = [upload(dev, ww[:, :, i : i + 1]) for i in range(nq)]

            def stock_weighted():
                return tuple(ttnn.multiply(stock_rms(v), w, dtype=BF) for v, w in zip(stock_mix(), individual))

            yield "mix3_weighted", lambda: rowvm.attnres_mix(s, p, q_rows, j, weight=weights, dtype=BF), stock_weighted

    for h, d, threshold in ((24, 16, 8.0), (24, 128, 22.627416997969522)):
        x = rnd((1, 1, h, d), BF)
        tau = upload(dev, torch.full((1, 1, 1, 1), 1.08))

        def stock_head_clip():
            xf = f32(x)
            ss = ttnn.sum(ttnn.multiply(xf, xf), dim=-1, keepdim=True)
            fac = [
                act("SQRT"),
                act("ADD_UNARY_SFPU", EPS),
                act("RECIP"),
                act("MUL_UNARY_SFPU", threshold),
                act("MINIMUM", 1.0),
            ]
            clip = ttnn.multiply(xf, ss, input_tensor_b_activations=fac, dtype=BF)
            return ttnn.multiply(f32(clip), tau, dtype=BF)

        yield f"head_clip{d}_tau", lambda: rowvm.head_norm(x, threshold=threshold, tau=tau), stock_head_clip

    x = rnd((1, 1, 8, 128), BF)

    def stock_head_rms():
        xf = f32(x)
        ss = ttnn.sum(ttnn.multiply(xf, xf), dim=-1, keepdim=True)
        fac = [act("MUL_UNARY_SFPU", 1.0 / 128), act("ADD_UNARY_SFPU", EPS), act("RSQRT")]
        return ttnn.multiply(xf, ss, input_tensor_b_activations=fac, dtype=BF)

    yield "head_rms128", lambda: rowvm.head_norm(x), stock_head_rms

    for width, dynamic in ((2304, False), (1024, True)):
        x = row(width, BF)
        taps = rnd((1, 1, 4, width))
        y0 = torch.zeros(1, 1, 4, width)
        state_a, state_b = upload(dev, y0), upload(dev, y0)
        e0 = upload(dev, torch.tensor([1, 0, 0, 0]).float().reshape(1, 1, 4, 1))
        shift_t = torch.zeros(1, 1, 4, 4)
        shift_t[0, 0, 1:, :3] = torch.eye(3)
        shift = upload(dev, shift_t)
        dyn = row(32, BF) if dynamic else None
        if dynamic:
            mask_t = torch.zeros(4, 32)
            expand_t = torch.zeros(32, width)
            for h in range(8):
                for lag in range(4):
                    mask_t[lag, h * 4 + lag] = 1
                    expand_t[h * 4 + lag, h * 128 : (h + 1) * 128] = 1 / 3
            ones = upload(dev, torch.ones(1, 1, 4, 1))
            lagmask = upload(dev, mask_t.reshape(1, 1, 4, 32))
            expand = upload(dev, expand_t.reshape(1, 1, 32, width))

        def stock_conv():
            ttnn.copy(ttnn.add(mm(shift, state_b), mm(e0, f32(x))), state_b)
            tap = taps
            if dynamic:
                tap = ttnn.add(tap, mm(ttnn.multiply(mm(ones, f32(dyn)), lagmask), expand))
            return ttnn.sum(ttnn.multiply(state_b, tap), dim=-2, keepdim=True)

        yield (
            f"conv{width}" + ("_dynamic" if dynamic else ""),
            (lambda: rowvm.depthwise_conv(state_a, x, taps, dyn=dyn)),
            stock_conv,
        )

    gu = rnd((1, 8, 1, 2304), BF, L1)

    def stock_hidden():
        g = ttnn.slice(gu, [0, 0, 0, 0], [1, 8, 1, 1152])
        u = ttnn.slice(gu, [0, 0, 0, 1152], [1, 8, 1, 2304])
        first = ttnn.multiply(
            g,
            g,
            input_tensor_a_activations=[act("SOFTCAP", 4.0)],
            input_tensor_b_activations=[act("SIGMOID")],
            dtype=FP,
        )
        return ttnn.multiply(first, u, input_tensor_b_activations=[act("SOFTCAP", 25.0)], dtype=BF)

    yield "moe_hidden", lambda: rowvm.moe_hidden(gu), stock_hidden
    d, gamma = rnd((1, 8, 1, D), BF, L1), row(8)

    def stock_combine():
        a = ttnn.reshape(d, (1, 1, 8, D))
        return ttnn.matmul(gamma, a, compute_kernel_config=ck, dtype=BF)

    yield "moe_combine", lambda: rowvm.moe_combine(d, gamma), stock_combine

    x, g = row(3072, BF), row(3072)
    yield (
        "sigmoid_gate",
        lambda: rowvm.sigmoid_gate(x, g),
        lambda: ttnn.multiply(x, g, input_tensor_b_activations=[act("SIGMOID")], dtype=BF),
    )
    yield "residual_add", lambda: rowvm.add(x, g), lambda: bf16(ttnn.add(f32(x), g))

    u, m, f, g, w = (row() for _ in range(5))

    def stock_msg_output():
        a = ttnn.multiply(u, f, input_tensor_b_activations=[act("TANH"), act("RSUB", 1.0)])
        b = ttnn.multiply(m, g, input_tensor_b_activations=[act("TANH")])
        return ttnn.multiply(stock_rms(ttnn.add(a, b)), w, dtype=BF)

    yield (
        "message_input",
        lambda: rowvm.message_input(u, m),
        lambda: bf16(ttnn.concat([stock_rms(u), stock_rms(m)], -1)),
    )
    yield "message_output", lambda: rowvm.message_output(u, m, f, g, weight=w, dtype=BF), stock_msg_output

    logits, bias = row(128), row(128)

    def stock_router():
        s = ttnn.sigmoid(logits)
        _, ids = ttnn.topk(ttnn.add(s, bias), k=8, dim=-1, largest=True, sorted=True)
        picked = ttnn.gather(s, -1, ids)
        denom = ttnn.add(ttnn.sum(picked, dim=-1, keepdim=True), 1e-20)
        gamma = ttnn.multiply(ttnn.multiply(picked, ttnn.reciprocal(denom)), 8.0)
        return ttnn.to_layout(ttnn.typecast(ids, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT), gamma

    yield "router", lambda: rowvm.router(logits, bias), stock_router


def profile(dev, fn, name, variant, calls, replays):
    last = fn()
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    trace = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(calls):
        last = fn()
    ttnn.end_trace_capture(dev, trace, cq_id=0)
    ttnn.synchronize_device(dev)
    durations = []
    for _ in range(replays):
        start = time.perf_counter()
        ttnn.execute_trace(dev, trace, cq_id=0, blocking=True)
        durations.append((time.perf_counter() - start) * 1e6 / calls)
        ttnn.ReadDeviceProfiler(dev)
    ttnn.release_trace(dev, trace)
    # Keep the final captured output allocation live until replay is done.
    del last
    print(
        "ROWVM_PROFILE "
        + json.dumps(
            dict(case=name, variant=variant, trace=int(trace), calls=calls, host_us=statistics.median(durations))
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append")
    parser.add_argument("--variant", action="append", choices=("rowvm", "stock"))
    parser.add_argument("--calls", type=int, default=16)
    parser.add_argument("--replays", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(17)
    dev = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=32_000_000)
    try:
        dev.enable_program_cache()
        for name, fused, stock in cases(dev):
            if args.case is None or name in args.case:
                for variant, fn in (("rowvm", fused), ("stock", stock)):
                    if args.variant is None or variant in args.variant:
                        profile(dev, fn, name, variant, args.calls, args.replays)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
