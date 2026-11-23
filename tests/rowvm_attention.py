"""Verify attention glue and profile the trace chains used by Layer.attention."""

import argparse
from functools import partial
from itertools import chain

import torch
import ttnn
from bench_rowvm import profile
from test_rowvm import BF, DRAM, FP, L1, upload

from inkling_tt import rowvm
from inkling_tt.model import DR, HD, HKV, HQ, NEG, QK_CLIP, REL_CLIP, W
from inkling_tt.model2 import Ctx, Ops, Precision

PACKED_WIDTH = 2 * HQ * HD + HQ * DR + 32
R_COL = HQ * HD
SINK_COL = R_COL + HQ * DR
GATE_COL = SINK_COL + 32


def cpu(tensor):
    return ttnn.to_torch(tensor)


def bitwise(actual, expected):
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    got, want = actual.view(torch.int16), expected.view(torch.int16)
    mismatches = (got != want).count_nonzero().item()
    assert mismatches == 0, f"{mismatches} mismatched BF16 values in {actual.shape}"


def relerr(actual, expected):
    actual, expected = actual.float(), expected.float()
    error = ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()
    assert error <= 1e-5, error
    return error


def take(x, start, width):
    return ttnn.slice(x, [0, 0, 0, start], [1, 1, 1, start + width], memory_config=x.memory_config())


def stock_heads(row, col, heads, dim, threshold=None, tau=None, dtype=BF, pad=False):
    if col is not None:
        row = take(row, col, heads * dim)
    x = ttnn.reshape(row, (1, 1, heads, dim))
    y = rowvm.head_norm(x, threshold=threshold, tau=tau, dtype=dtype)
    return ttnn.pad(y, [(0, 0), (0, 0), (0, 0), (0, 16)], 0.0) if pad else y


def stock_sink(row, col):
    if col is not None:
        row = take(row, col, HQ)
    heads = ttnn.reshape(row, (1, 1, HQ, 1))
    return ttnn.reshape(ttnn.pad(heads, [(0, 0), (0, 0), (0, 32 - HQ), (0, 31)], 0.0), (32, 32))


def stock_gate(x, row, col, dtype=BF):
    if col is not None:
        row = take(row, col, HQ * HD)
    return rowvm.sigmoid_gate(ttnn.reshape(x, (1, 1, 1, HQ * HD)), row, dtype=dtype)


def stock_kv(row, shard, threshold=None):
    return ttnn.to_memory_config(stock_heads(row, None, HKV, HD, threshold), shard)


def stock_bias(ctx, r, idx, table, mask, dtype=BF):
    rel = ttnn.embedding(idx, table, layout=ttnn.TILE_LAYOUT, dtype=BF)
    rel = ttnn.reshape(rel, (1, 1, idx.shape[-1], 32))
    return ttnn.add(ctx.mm(r, rel, transpose_b=True), mask, dtype=dtype)


def bias_inputs(ctx, width, *, arbitrary=False):
    dim = 32 if arbitrary else DR
    r = torch.nn.functional.pad(torch.randn(1, 1, HQ, dim), (0, 32 - dim))
    table = torch.zeros(W + 1, 32)
    table[:W, :dim] = torch.randn(W, dim) * HD**0.5
    if arbitrary:
        ids = torch.randint(0, W + 1, (width,), dtype=torch.int32)
        ids[:32] = ids[32] = ids[63] = 7
        mask = torch.randn(width) * 10
    else:
        dist = 733 - torch.arange(width, dtype=torch.int32)
        if width == W:
            ids = dist % W
            valid = ids <= 733
        else:
            valid = dist >= 0
            ids = torch.where(valid, dist, W)
        mask = torch.where(valid, 0.0, NEG)
    return (
        ctx.up(r),
        ctx.up(ids.reshape(1, width), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        ctx.up(table, BF, ttnn.ROW_MAJOR_LAYOUT),
        ctx.up(mask.reshape(1, 1, 1, width), FP),
    )


def bias_cases(ctx):
    for width in (1024, 2048):
        inputs = bias_inputs(ctx, width)
        yield f"relative_{width}", partial(rowvm.relative_bias, *inputs), partial(stock_bias, ctx, *inputs)


def cases(dev, shard, mem=DRAM):
    packed = upload(dev, torch.randn(1, 1, 1, PACKED_WIDTH) * 3, BF, mem)
    tau = upload(dev, torch.full((1, 1, 1, 1), 1.09))
    out = upload(dev, torch.randn(1, 1, HQ, HD), BF)
    k = upload(dev, torch.randn(1, 1, 1, HKV * HD) * 3, BF, mem)
    v = upload(dev, torch.randn(1, 1, 1, HKV * HD), BF, mem)

    for suffix, step_tau in (("local", None), ("global", tau)):
        yield (
            f"q_{suffix}",
            partial(rowvm.head_norm_from_row, packed, HQ, HD, threshold=QK_CLIP, tau=step_tau),
            partial(stock_heads, packed, 0, HQ, HD, QK_CLIP, step_tau),
        )
        yield (
            f"r_{suffix}",
            partial(
                rowvm.head_norm_from_row,
                packed,
                HQ,
                DR,
                col_offset=R_COL,
                threshold=REL_CLIP,
                tau=step_tau,
                out_shape=(1, 1, HQ, 32),
            ),
            partial(stock_heads, packed, R_COL, HQ, DR, REL_CLIP, step_tau, pad=True),
        )
    yield (
        "k_shard",
        partial(rowvm.kv_norm_to_shard, k, threshold=QK_CLIP, memory_config=shard),
        partial(
            stock_kv,
            k,
            shard,
            QK_CLIP,
        ),
    )
    yield "v_shard", partial(rowvm.kv_norm_to_shard, v, memory_config=shard), partial(stock_kv, v, shard)
    yield "sink", partial(rowvm.sink_tile, packed, col_offset=SINK_COL), partial(stock_sink, packed, SINK_COL)
    yield (
        "gate",
        partial(rowvm.gate_from_heads, out, packed, col_offset=GATE_COL),
        partial(stock_gate, out, packed, GATE_COL),
    )


def verify_outputs(dev, ctx):
    for mem in (DRAM, L1):
        # The packed projection and the conv outputs follow the input placement.
        for name, fused, stock in cases(dev, ctx.kv_shard, mem):
            actual, expected = cpu(fused()), cpu(stock())
            bitwise(actual, expected)
            print(f"{name} {mem.buffer_type}: BF16 bitwise", flush=True)

    kv_row = upload(dev, torch.randn(1, 1, 1, HKV * HD) * 3)
    for threshold in (QK_CLIP, None):
        actual = rowvm.kv_norm_to_shard(kv_row, threshold=threshold, memory_config=ctx.kv_shard)
        bitwise(cpu(actual), cpu(stock_kv(kv_row, ctx.kv_shard, threshold)))
    print("K and V shards from FP32 conv rows: BF16 bitwise")

    # Offset by one face to exercise both halves of a tile and cross a tile
    # boundary for the sink. A negative tau must still leave positive zero
    # in the relative head's extension.
    x = torch.randn(1, 1, 1, 16 + PACKED_WIDTH) * 3
    row = upload(dev, x)
    row_bf = upload(dev, x, BF)
    tau = upload(dev, torch.full((1, 1, 1, 1), -1.09))
    for dim, col, threshold in ((HD, 16, QK_CLIP), (DR, 16 + R_COL, REL_CLIP)):
        shape = (1, 1, HQ, dim)
        step_out = (1, 1, HQ, 32) if dim == DR else shape
        heads = upload(dev, x[..., col : col + HQ * dim].reshape(shape))
        reference = rowvm.head_norm(heads, threshold=threshold, tau=tau, dtype=FP)
        out = rowvm.head_norm_from_row(
            row, HQ, dim, out_shape=step_out, threshold=threshold, tau=tau, col_offset=col, dtype=FP
        )
        got = cpu(out)
        if dim == DR:
            assert (got[..., DR:].view(torch.int32) == 0).all()
        print(f"head clip {dim} with tau vs reshape+norm FP32 relerr {relerr(got[..., :dim], cpu(reference)):.3g}")
        reference = Ops.hclip(None, heads, threshold, out=FP)
        out = rowvm.head_norm_from_row(row, HQ, dim, threshold=threshold, col_offset=col, dtype=FP)
        print(f"head clip {dim} vs TTNN reduction FP32 relerr {relerr(cpu(out), cpu(reference)):.3g}")

    heads = upload(dev, x[..., 16 : 16 + HKV * HD].reshape(1, 1, HKV, HD))
    for name, threshold, reference in (
        ("L2", QK_CLIP, Ops.hclip(None, heads, QK_CLIP)),
        ("RMS", None, Ops.hrms(None, heads)),
    ):
        out = rowvm.head_norm_from_row(row, HKV, HD, threshold=threshold, col_offset=16, dtype=FP)
        print(f"KV {name} math vs TTNN reduction FP32 relerr {relerr(cpu(out), cpu(reference)):.3g}")
    logits = torch.randn(1, 1, HQ, HD).bfloat16()
    values = upload(dev, logits, BF)
    got = rowvm.gate_from_heads(values, row, col_offset=16 + GATE_COL, dtype=FP)
    gate = upload(dev, x[..., 16 + GATE_COL : 16 + GATE_COL + HQ * HD])
    ref = stock_gate(values, gate, None, dtype=FP)
    print(f"gate vs reshape+gate FP32 relerr {relerr(cpu(got), cpu(ref)):.3g}")
    expected = torch.zeros(32, 32, dtype=torch.bfloat16)
    expected[:HQ, 0] = x.flatten()[16 + SINK_COL : 16 + SINK_COL + HQ]
    sink = rowvm.sink_tile(row_bf, col_offset=16 + SINK_COL)
    assert sink.memory_config() == DRAM
    bitwise(cpu(sink), expected)
    bitwise(cpu(rowvm.sink_tile(upload(dev, expected[:HQ, :1].reshape(1, 1, 1, HQ), BF))), expected)
    print("sink from standalone and face-offset rows: BF16 bitwise", flush=True)


def verify_replay(dev, ctx, klen):
    is_global = klen != W
    packed = upload(dev, torch.zeros(1, 1, 1, PACKED_WIDTH), BF)
    tau = upload(dev, torch.ones(1, 1, 1, 1)) if is_global else None
    rows = [upload(dev, torch.zeros(1, 1, 1, HKV * HD), BF) for _ in range(2)]
    cache_shape = (1, HKV, klen, HD)
    caches = [upload(dev, torch.zeros(cache_shape), BF) for _ in range(2)]
    stock_caches = [upload(dev, torch.zeros(cache_shape), BF) for _ in range(2)]
    thresholds = (QK_CLIP, None)
    pos = ttnn.from_torch(
        torch.zeros(1, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=DRAM,
    )
    idx = ttnn.from_torch(
        torch.zeros(1, klen, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=DRAM,
    )
    table_data = torch.zeros(W + 1, 32)
    table_data[:W, :DR] = torch.randn(W, DR)
    table = ttnn.from_torch(table_data, dtype=BF, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=DRAM)
    mask = upload(dev, torch.zeros(1, 1, 1, klen))

    def sdpa(q, kv, sink, bias):
        return ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            *kv,
            is_causal=False,
            attn_mask=bias,
            attention_sink=sink,
            scale=HD**-0.5,
            program_config=ctx.sdpa_pc,
            compute_kernel_config=ctx.ck_exact,
            memory_config=DRAM,
        )

    q_fn = partial(rowvm.head_norm_from_row, packed, HQ, HD, threshold=QK_CLIP, tau=tau)
    r_fn = partial(
        rowvm.head_norm_from_row,
        packed,
        HQ,
        DR,
        col_offset=R_COL,
        threshold=REL_CLIP,
        tau=tau,
        out_shape=(1, 1, HQ, 32),
    )
    sink_fn = partial(rowvm.sink_tile, packed, col_offset=SINK_COL)
    kv_fns = [
        partial(rowvm.kv_norm_to_shard, row, threshold=threshold, memory_config=ctx.kv_shard)
        for row, threshold in zip(rows, thresholds)
    ]
    q, r, sink, *kvs = [fn() for fn in (q_fn, r_fn, sink_fn, *kv_fns)]
    bias = rowvm.relative_bias(r, idx, table, mask)
    gated = rowvm.gate_from_heads(sdpa(q, caches, sink, bias), packed, col_offset=GATE_COL)
    assert all(tuple(kv.padded_shape) == (1, 1, 32, 128) and kv.memory_config() == ctx.kv_shard for kv in kvs)

    def fused():
        q_fn(out=q)
        r_fn(out=r)
        sink_fn(out=sink)
        for cache, fn, kv in zip(caches, kv_fns, kvs):
            ttnn.experimental.paged_update_cache(cache, fn(out=kv), update_idxs_tensor=pos)
        rowvm.relative_bias(r, idx, table, mask, out=bias)
        return rowvm.gate_from_heads(sdpa(q, caches, sink, bias), packed, col_offset=GATE_COL, out=gated)

    def reference_step():
        for cache, row, threshold in zip(stock_caches, rows, thresholds):
            ttnn.experimental.paged_update_cache(cache, stock_kv(row, ctx.kv_shard, threshold), update_idxs_tensor=pos)
        q_ref = stock_heads(packed, 0, HQ, HD, QK_CLIP, tau)
        r_ref = stock_heads(packed, R_COL, HQ, DR, REL_CLIP, tau, pad=True)
        sink_ref = stock_sink(packed, SINK_COL)
        bias_ref = stock_bias(ctx, r_ref, idx, table, mask)
        gate_ref = stock_gate(sdpa(q_ref, stock_caches, sink_ref, bias_ref), packed, GATE_COL)
        return [cpu(t) for t in (*stock_caches, q_ref, r_ref, sink_ref, bias_ref, gate_ref)]

    steps = []
    for p in (3, 35, W + 3):
        distance = p - torch.arange(klen, dtype=torch.int32)
        if is_global:
            valid = distance >= 0
            ids = torch.where(valid & (distance < W), distance, W)
        else:
            ids = distance % W
            valid = ids <= p
        step_values = [
            (packed, torch.randn(1, 1, 1, PACKED_WIDTH) * 3),
            (pos, torch.tensor([p if is_global else p % W], dtype=torch.int32)),
            (idx, ids.reshape(1, klen)),
            (mask, torch.where(valid, 0.0, NEG).reshape(1, 1, 1, klen)),
            *((row, torch.randn(1, 1, 1, HKV * HD) * 3) for row in rows),
        ]
        if is_global:
            step_values.append((tau, torch.full((1, 1, 1, 1), 1 + p / W)))
        inputs = [
            (ttnn.from_torch(values, dtype=tensor.dtype, layout=tensor.layout), tensor)
            for tensor, values in step_values
        ]
        for host, tensor in inputs:
            ttnn.copy_host_to_device_tensor(host, tensor)
        steps.append((p, inputs, reference_step()))

    # Warm all cache-update and SDPA programs at a separate position, then
    # restore empty caches before replaying the reference positions.
    zero_pos = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), dtype=pos.dtype, layout=pos.layout)
    empty_cache = ttnn.from_torch(torch.zeros(cache_shape), dtype=BF, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(zero_pos, pos)
    fused()
    ttnn.synchronize_device(dev)
    trace = ttnn.begin_trace_capture(dev, cq_id=0)
    fused()
    ttnn.end_trace_capture(dev, trace, cq_id=0)
    try:
        for cache in caches:
            ttnn.copy_host_to_device_tensor(empty_cache, cache)
        for p, inputs, reference in steps:
            for host, tensor in inputs:
                ttnn.copy_host_to_device_tensor(host, tensor)
            ttnn.execute_trace(dev, trace, cq_id=0, blocking=True)
            for actual, expected in zip((*caches, q, r, sink, bias, gated), reference):
                bitwise(cpu(actual), expected)
            print(f"trace L={klen} pos={p}: cache, Q, R, sink, bias and gated SDPA match bitwise", flush=True)
    finally:
        ttnn.release_trace(dev, trace)


def verify_bias(ctx):
    for width in (1024, 2048, 4352):
        inputs = bias_inputs(ctx, width, arbitrary=True)
        actual = cpu(rowvm.relative_bias(*inputs, dtype=FP))
        expected = cpu(stock_bias(ctx, *inputs, dtype=FP))
        print(f"relative bias L={width}, arbitrary mask, FP32 relerr {relerr(actual, expected):.3g}", flush=True)
        inputs = bias_inputs(ctx, width)
        actual, expected = cpu(rowvm.relative_bias(*inputs)), cpu(stock_bias(ctx, *inputs))
        unequal = actual.view(torch.int16) != expected.view(torch.int16)
        if unequal.any():
            fp_actual = cpu(rowvm.relative_bias(*inputs, dtype=FP))
            fp_expected = cpu(stock_bias(ctx, *inputs, dtype=FP))
            print(
                "BF16 actual, BF16 stock, FP32 actual, FP32 stock at mismatches:",
                torch.stack(
                    [t[unequal][:8].float() for t in (actual, expected, fp_actual, fp_expected)], dim=1
                ).tolist(),
                flush=True,
            )
        bitwise(actual, expected)
        print(f"relative bias L={width}, decode mask: BF16 bitwise", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--bias-only", action="store_true")
    parser.add_argument("--case", action="append")
    parser.add_argument("--calls", type=int, default=24)
    parser.add_argument("--replays", type=int, default=5)
    args = parser.parse_args()
    torch.manual_seed(72)
    dev = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=32_000_000)
    try:
        dev.enable_program_cache()
        ctx = Ctx(dev, Precision(dense_fidelity=ttnn.MathFidelity.HiFi4, expert_fidelity=ttnn.MathFidelity.HiFi4))
        if args.bench:
            attention_cases = () if args.bias_only else cases(dev, ctx.kv_shard)
            for name, fused, stock in chain(attention_cases, bias_cases(ctx)):
                if args.case is None or name in args.case:
                    for variant, fn in (("rowvm", partial(fused, out=fused())), ("stock", stock)):
                        profile(dev, fn, name, variant, args.calls, args.replays)
        else:
            if not args.bias_only:
                verify_outputs(dev, ctx)
                for width in (W, 2 * W):
                    verify_replay(dev, ctx, width)
            verify_bias(ctx)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
