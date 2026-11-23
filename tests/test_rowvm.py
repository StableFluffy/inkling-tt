"""Correctness and replay checks for rowvm on the Blackhole card selected by TT_VISIBLE_DEVICES."""

import argparse
import math

import torch
import ttnn

from inkling_tt import rowvm

FP, BF = ttnn.float32, ttnn.bfloat16
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def rms(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


def upload(dev, data, dtype=FP, mem=DRAM):
    return ttnn.from_torch(data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem)


def check(name, actual, expected, fp32=None):
    actual = ttnn.to_torch(actual).float()
    expected = expected.float()
    error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected).clamp_min(1e-20)
    if fp32 is None:
        print(f"{name}: relerr={error:.3g}", flush=True)
        assert error <= 1e-4, (name, actual.flatten()[:16], expected.flatten()[:16])
    else:
        unrounded = ttnn.to_torch(fp32).float()
        rounded = unrounded.bfloat16().float()
        near_midpoints = torch.count_nonzero(rounded != expected).item()
        print(f"{name}: relerr={error:.3g}, torch rounding differences={near_midpoints}", flush=True)
        mismatch = actual != rounded
        assert not mismatch.any(), (name, unrounded[mismatch], actual[mismatch], rounded[mismatch])
        expected_bf16 = expected.bfloat16()
        lower = torch.nextafter(expected_bf16, torch.full_like(expected_bf16, -torch.inf)).float()
        upper = torch.nextafter(expected_bf16, torch.full_like(expected_bf16, torch.inf)).float()
        assert torch.all((lower <= actual) & (actual <= upper)), name


def test_rms(dev):
    for width, dtype, mem in (
        (32, FP, DRAM),
        (128, BF, DRAM),
        (1008, FP, L1),
        (1024, BF, L1),
        (2304, BF, DRAM),
        (4176, BF, DRAM),
        (4608, FP, DRAM),
    ):
        x, w = (torch.randn(1, 1, 1, width) for _ in range(2))
        if dtype == BF:
            x = x.bfloat16().float()
        a, b = upload(dev, x, dtype, mem), upload(dev, w)
        gold = rms(x) * w
        actual = rowvm.rms_norm(a, b)
        check(f"rms {width} {dtype}", actual, gold)
        check(f"rms bf16 {width}", rowvm.rms_norm(a, b, dtype=BF), gold.bfloat16(), actual)
        check(f"rms unweighted {width}", rowvm.rms_norm(a), rms(x))

    shape = (1, 1, 1, 2304)
    x = torch.randn(shape).bfloat16()
    a = upload(dev, x, BF)
    y = rowvm.rms_norm(a)
    ttnn.synchronize_device(dev)
    alternate = upload(dev, torch.zeros(shape), BF)
    y2 = rowvm.rms_norm(alternate)
    check("rms cache rebind", y2, torch.zeros(shape))
    trace = ttnn.begin_trace_capture(dev, cq_id=0)
    rowvm.rms_norm(a, out=y)
    ttnn.end_trace_capture(dev, trace, cq_id=0)
    try:
        for index in range(2):
            host = torch.randn(shape).bfloat16()
            origin = ttnn.from_torch(host, dtype=BF, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(origin, a)
            ttnn.execute_trace(dev, trace, cq_id=0, blocking=True)
            check(f"rms trace replay {index}", y, rms(host.float()))
    finally:
        ttnn.release_trace(dev, trace)


def test_heads(dev):
    for h, d, threshold in ((24, 16, 8.0), (24, 128, 22.627416997969522), (8, 128, 2.0)):
        x = torch.randn(1, 1, h, d).bfloat16().float() * 4
        x[:, :, 0] = 0
        gain = torch.randn_like(x)
        tau = torch.full((1, 1, 1, 1), 1.08)
        a, b, t = upload(dev, x, BF), upload(dev, gain), upload(dev, tau)
        scaled = x * gain
        clipped = scaled * (threshold / (scaled.norm(dim=-1, keepdim=True) + 1e-6)).clamp_max(1)
        args = dict(scale=b, threshold=threshold, tau=t)
        gold = clipped.bfloat16().float() * tau
        actual = rowvm.head_norm(a, dtype=FP, **args)
        check(f"head clip {h}x{d}", actual, gold)
        check(f"head clip BF16 {h}x{d}", rowvm.head_norm(a, **args), gold.bfloat16(), actual)
        actual = rowvm.head_norm(a, dtype=FP)
        check(f"head RMS {h}x{d}", actual, rms(x))
        check(f"head RMS BF16 {h}x{d}", rowvm.head_norm(a), rms(x).bfloat16(), actual)


def test_conv(dev):
    for width, dynamic in ((2304, False), (1024, True)):
        taps = torch.randn(1, 1, 4, width) / math.sqrt(5)
        state = torch.randn_like(taps)
        s, t = upload(dev, state), upload(dev, taps)
        dyn = torch.randn(1, 1, 1, 32) if dynamic else None
        args = dict(dyn=upload(dev, dyn, BF) if dynamic else None)
        taps_eff = taps
        if dynamic:
            broadcast = dyn.bfloat16().float().reshape(8, 4).T.repeat_interleave(128, dim=-1)
            taps_eff = taps + broadcast / 3
        for step in range(2):
            x = torch.randn(1, 1, 1, width).bfloat16().float()
            a = upload(dev, x, BF)
            state = torch.cat([x, state[:, :, :3]], dim=-2)
            gold = (state * taps_eff).sum(-2, keepdim=True)
            check(f"conv dynamic={dynamic} step={step}", rowvm.depthwise_conv(s, a, t, **args), gold)
            check(f"conv state dynamic={dynamic} step={step}", s, state)

    # One persistent static history and row are updated through the same captured launch.
    width = 2304
    shape = (1, 1, 1, width)
    taps_host = torch.randn(1, 1, 4, width)
    state_host = torch.zeros_like(taps_host)
    s, sf, taps = (upload(dev, v) for v in (state_host, state_host, taps_host))
    a = upload(dev, torch.zeros(shape), BF)
    y = rowvm.depthwise_conv(s, a, taps, dtype=BF)
    ttnn.synchronize_device(dev)
    trace = ttnn.begin_trace_capture(dev, cq_id=0)
    rowvm.depthwise_conv(s, a, taps, dtype=BF, out=y)
    ttnn.end_trace_capture(dev, trace, cq_id=0)
    ttnn.synchronize_device(dev)
    try:
        for step in range(3):
            host_x = torch.randn(shape).bfloat16()
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(host_x, dtype=BF, layout=ttnn.TILE_LAYOUT), a)
            ttnn.execute_trace(dev, trace, cq_id=0, blocking=True)
            state_host = torch.cat([host_x.float(), state_host[:, :, :3]], dim=-2)
            torch_y = (state_host * taps_host).sum(-2, keepdim=True)
            fp = rowvm.depthwise_conv(sf, a, taps, dtype=FP)
            check(f"conv BF16 trace step={step}", y, torch_y.bfloat16(), fp)
            check(f"conv replayed state step={step}", s, state_host)
    finally:
        ttnn.release_trace(dev, trace)


def test_moe(dev):
    gu = (torch.randn(1, 8, 1, 2304) * 9).bfloat16().float()
    g, u = gu.split(1152, dim=-1)
    gold = (4 * (g / 4).tanh()) * g.sigmoid() * (25 * (u / 25).tanh())
    for shape in ((1, 8, 1, 2304), (1, 1, 8, 2304)):
        a = upload(dev, gu.reshape(shape), BF, L1)
        reference = gold.reshape(*shape[:-1], 1152)
        actual = rowvm.moe_hidden(a, dtype=FP)
        check(f"moe hidden {shape}", actual, reference)
        check(f"moe hidden BF16 {shape}", rowvm.moe_hidden(a), reference.bfloat16(), actual)
    d = torch.randn(1, 8, 1, 2304).bfloat16().float()
    gamma = torch.rand(1, 1, 1, 8)
    a, g = upload(dev, d, BF, L1), upload(dev, gamma)
    gold = (d * gamma.reshape(1, 8, 1, 1)).sum(1, keepdim=True)
    actual = rowvm.moe_combine(a, g, dtype=FP)
    check("moe combine", actual, gold)
    check("moe combine BF16", rowvm.moe_combine(a, g), gold.bfloat16(), actual)


def test_pointwise(dev):
    x = torch.randn(1, 1, 1, 3072).bfloat16().float()
    logits = torch.randn_like(x) * 4
    a, b = upload(dev, x, BF), upload(dev, logits)
    for name, fn, gold in (
        ("add", rowvm.add, x + logits),
        ("sigmoid gate", rowvm.sigmoid_gate, x * logits.sigmoid()),
    ):
        actual = fn(a, b, dtype=FP)
        check(name, actual, gold)
        check(name + " BF16", fn(a, b), gold.bfloat16(), actual)

    x = torch.tensor([1.0, 1.0078125, -1.0, -1.0078125]).repeat(8).reshape(1, 1, 1, 32)
    delta = torch.tensor([1, 1, -1, -1]).repeat(8).reshape_as(x).float() / 256
    a, b = upload(dev, x, BF), upload(dev, delta)
    out = rowvm.add(a, b)
    fp = rowvm.add(a, b, dtype=FP)
    check("add BF16 ties-to-even", out, (x + delta).bfloat16(), fp)


def test_message(dev):
    u, m, f, g, w = (torch.randn(1, 1, 1, 2304) for _ in range(5))
    a, b, F, G, W = (upload(dev, v) for v in (u, m, f, g, w))
    blend = (1 - f.tanh()) * u + g.tanh() * m
    check("message blend", rowvm.message_output(a, b, F, G), blend)
    cat = torch.cat([rms(u), rms(m)], -1)
    for name, args in (("message input", {}), ("message input pre-normalized", {"normalized_u": upload(dev, rms(u))})):
        actual = rowvm.message_input(a, b, dtype=FP, **args)
        check(name, actual, cat)
        check(name + " BF16", rowvm.message_input(a, b, **args), cat.bfloat16(), actual)
    actual = rowvm.message_output(a, b, F, G, weight=W)
    gold = rms(blend) * w
    check("message weighted", actual, gold)
    check("message weighted BF16", rowvm.message_output(a, b, F, G, weight=W, dtype=BF), gold.bfloat16(), actual)


def test_mix(dev):
    for width, j, n, start in ((256, 0, 1, 0), (2304, 4, 3, 0), (2304, 7, 4, 0), (128, 32, 2, 0), (672, 6, 2, 14)):
        bank = rms(torch.randn(1, 1, 32, width))
        bank[:, :, :start] = float("nan")
        bank[:, :, start + j :] = float("nan")
        partial = torch.randn(1, 1, 1, width).bfloat16().float()
        queries = torch.randn(1, 1, n, width) / math.sqrt(width)
        if start:
            queries *= 200
        weight = torch.randn_like(queries)
        samples = torch.cat([bank[:, :, start : start + j], rms(partial)], dim=-2)
        mix = (queries @ samples.mT).softmax(-1) @ samples
        inputs = upload(dev, bank), upload(dev, partial, BF), upload(dev, queries), j
        actual = rowvm.attnres_mix(*inputs, bank_start=start)
        for i, out in enumerate(actual):
            check(f"mix j={j} n={n} q={i}", out, mix[:, :, i : i + 1])
        if n == 3:
            w = upload(dev, weight)
            gold = rms(mix) * weight
            actual = rowvm.attnres_mix(*inputs, weight=w)
            rounded = rowvm.attnres_mix(*inputs, weight=w, dtype=BF)
            for i, (f, b) in enumerate(zip(actual, rounded)):
                expected = gold[:, :, i : i + 1]
                check(f"weighted mix q={i}", f, expected)
                check(f"weighted mix BF16 q={i}", b, expected.bfloat16(), f)

            ttnn.synchronize_device(dev)
            trace = ttnn.begin_trace_capture(dev, cq_id=0)
            rowvm.attnres_mix(*inputs, weight=w, dtype=BF, out=rounded)
            ttnn.end_trace_capture(dev, trace, cq_id=0)
            try:
                new_partial = torch.randn(1, 1, 1, width).bfloat16()
                host = ttnn.from_torch(new_partial, dtype=BF, layout=ttnn.TILE_LAYOUT)
                ttnn.copy_host_to_device_tensor(host, inputs[1])
                ttnn.execute_trace(dev, trace, cq_id=0, blocking=True)
                new_samples = torch.cat([bank[:, :, :j], rms(new_partial.float())], dim=-2)
                new_mix = (queries @ new_samples.mT).softmax(-1) @ new_samples
                expected = rms(new_mix) * weight
                reference = rowvm.attnres_mix(*inputs, weight=w)
                for i, (f, b) in enumerate(zip(reference, rounded)):
                    check(f"weighted mix replay q={i}", b, expected[:, :, i : i + 1].bfloat16(), f)
            finally:
                ttnn.release_trace(dev, trace)


def test_router(dev):
    for bias_enabled, mem in ((True, DRAM), (False, L1)):
        logits, bias = (torch.randn(1, 1, 1, 128) for _ in range(2))
        if not bias_enabled:
            bias.zero_()
        a, b = upload(dev, logits), upload(dev, bias)
        ids, gamma = rowvm.router(a, b, memory_config=mem)
        score = logits.sigmoid()
        expected_ids = (score + bias).topk(8, dim=-1).indices
        picked = score.gather(-1, expected_ids)
        assert torch.equal(ttnn.to_torch(ids).long(), expected_ids)
        check(f"router gamma bias={bias_enabled}", gamma, 8 * picked / (picked.sum(-1, keepdim=True) + 1e-20))

    logits = upload(dev, torch.zeros(1, 1, 1, 128))
    ids, gamma = rowvm.router(logits, logits)
    assert torch.equal(ttnn.to_torch(ids).long().reshape(-1), torch.arange(8))
    check("router ties", gamma, torch.ones(1, 1, 1, 8))

    # The sparse matmul must accept the generated stick directly.
    values = torch.arange(1, 129, dtype=torch.float32).view(1, 128, 1, 1)
    matrices = values * torch.eye(32).view(1, 1, 32, 32)
    inp = upload(dev, torch.ones(1, 1, 1, 32), BF, L1)
    weights = upload(dev, matrices, BF)
    dummy = ttnn.from_torch(
        torch.zeros(1, 1, 1, 128), dtype=BF, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=L1
    )
    program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    gu = ttnn.sparse_matmul(
        inp,
        weights,
        sparsity=dummy,
        indices=ids,
        program_config=program,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        dtype=BF,
        memory_config=L1,
    )
    got = ttnn.to_torch(gu).reshape(8, 32).float()
    assert torch.equal(got, values.flatten()[:8, None].expand(8, 32))
    print("router sparse_matmul stick accepted", flush=True)


CASES = dict(
    rms=test_rms,
    heads=test_heads,
    conv=test_conv,
    moe=test_moe,
    pointwise=test_pointwise,
    message=test_message,
    mix=test_mix,
    router=test_router,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(72)
    dev = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=8_000_000)
    try:
        dev.enable_program_cache()
        failed = []
        for case in args.case or CASES:
            try:
                CASES[case](dev)
            except Exception:
                if not args.keep_going:
                    raise
                import traceback

                traceback.print_exc()
                failed.append(case)
        if failed:
            raise SystemExit(f"Failed cases: {', '.join(failed)}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
