"""Compare packed block prefill with stock sparse prefill and per-token decode."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch
import ttnn
from safetensors import safe_open

from inkling_tt import experts_stream, paths
from inkling_tt.model2 import act

D, F, E, K = 2304, 1152, 128, 8
BF, FP, TILE, RM = ttnn.bfloat16, ttnn.float32, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
DRAM = ttnn.DRAM_MEMORY_CONFIG


def upload(device, x, dtype=BF, layout=TILE, memory=DRAM):
    return ttnn.from_torch(x.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=memory)


def route(x, rw, rb, valid):
    # Keep Prefill.moe's FP32 host routing, including topk order and normalization.
    scores = torch.sigmoid(x.reshape(-1, D)[:valid].float() @ rw)
    ids = torch.topk(scores + rb, K, dim=-1).indices
    gamma = scores.gather(1, ids)
    return ids, gamma / (gamma.sum(-1, keepdim=True) + 1e-20) * 8.0


def stock(device, gu, dn):
    w_gu = upload(device, gu.transpose(1, 2)[None], ttnn.bfloat8_b)
    w_dn = upload(device, dn.transpose(1, 2)[None], ttnn.bfloat8_b)
    dummy = upload(device, torch.zeros((1, 1, 1, E)), BF, RM)
    all_ids = upload(device, torch.arange(E, dtype=torch.int32).reshape(1, 1, 1, E), ttnn.uint16, RM)
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    def pc(tokens, kw):
        mt = tokens // 32
        sb = max(d for d in (1, 2, 4) if mt % d == 0)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(9, 8),
            in0_block_w=kw,
            out_subblock_h=sb,
            out_subblock_w=1,
            out_block_h=mt,
            out_block_w=1,
            per_core_M=mt,
            per_core_N=1,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def stage(routing, tokens):
        ids, gamma = routing
        g = torch.zeros(tokens, E)
        g[: ids.shape[0]].scatter_(1, ids, gamma)
        return upload(device, g.T.reshape(1, E, tokens, 1), FP)

    def step(x, gamma, *, fp32_epilogue=False):
        # The default is stock Prefill.moe after the route upload. The FP32
        # version tests the requested rounding boundaries using stock matmuls.
        tokens = x.shape[-2]
        common = dict(
            sparsity=dummy,
            indices=all_ids,
            is_input_b_sparse=True,
            memory_config=DRAM,
            dtype=BF,
            compute_kernel_config=ck,
        )
        z = ttnn.sparse_matmul(x, w_gu, is_input_a_sparse=False, program_config=pc(tokens, 8), **common)
        z = ttnn.reshape(z, (1, E, tokens, 2 * F))
        g = ttnn.slice(z, [0, 0, 0, 0], [1, E, tokens, F])
        u = ttnn.slice(z, [0, 0, 0, F], [1, E, tokens, 2 * F])
        if fp32_epilogue:
            g, u = ttnn.typecast(g, FP), ttnn.typecast(u, FP)
        h = ttnn.multiply(
            g,
            g,
            input_tensor_a_activations=[act("SOFTCAP", 4.0)],
            input_tensor_b_activations=[act("SIGMOID")],
            dtype=FP,
        )
        h = ttnn.multiply(h, u, input_tensor_b_activations=[act("SOFTCAP", 25.0)], dtype=BF)
        d = ttnn.sparse_matmul(h, w_dn, is_input_a_sparse=True, program_config=pc(tokens, 6), **common)
        d = ttnn.reshape(d, (1, E, tokens, D))
        weighted = ttnn.multiply(d, gamma, dtype=FP if fp32_epilogue else None)
        y = ttnn.sum(weighted, dim=1, keepdim=True)
        return y if y.dtype == BF else ttnn.typecast(y, BF)

    return stage, step


def emit(**record):
    print(json.dumps(record), flush=True)


def benchmark(device, step, repeats, reference=None):
    out = step()
    ttnn.synchronize_device(device)
    value = ttnn.to_torch(out) if reference is None else reference
    del out
    times = []
    for _ in range(repeats):
        start = perf_counter_ns()
        step()
        ttnn.synchronize_device(device)
        times.append((perf_counter_ns() - start) / 1e6)
    return value, median(times)


def metrics(actual, expected):
    delta = actual.float() - expected.float()
    return dict(
        equal_bits=int((actual.view(torch.int16) == expected.view(torch.int16)).sum()),
        elements=expected.numel(),
        rel_l2=float(delta.norm() / expected.float().norm()),
        max_abs=float(delta.abs().max()),
    )


def token_decode(device, x, routing, handle, repeats):
    ids, gamma = routing
    # The interim Prefill.moe stream passes DRAM tensors to moe_decode.
    # Pre-stage them to isolate the expert path in the timing.
    inputs = [
        (
            upload(device, xx.reshape(1, 1, 1, D)),
            upload(device, ii.reshape(1, 1, 1, K).int(), ttnn.uint16, RM),
            upload(device, gg.reshape(1, 1, 1, K), FP),
        )
        for xx, ii, gg in zip(x.reshape(-1, D), ids, gamma)
    ]

    def step():
        for case in inputs:
            experts_stream.moe_decode(*case, handle)

    value = torch.cat([ttnn.to_torch(experts_stream.moe_decode(*case, handle)) for case in inputs], dim=-2)
    _, elapsed = benchmark(device, step, repeats, reference=value)
    return value, elapsed


def trace_replay(device, x, routes, workspace, handle):
    expected = [
        ttnn.to_torch(experts_stream.moe_prefill(x, routing, handle, workspace=workspace)) for routing in routes
    ]
    ttnn.synchronize_device(device)
    # x, routes, output and scratch already have addresses. No staging or
    # first-time invocation occurs during capture.
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        y = experts_stream.moe_prefill(x, None, handle, workspace=workspace)
    finally:
        ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for routing, target in zip(routes, expected):
            experts_stream.stage_prefill_routing(routing, workspace)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            assert torch.equal(ttnn.to_torch(y).view(torch.int16), target.view(torch.int16))
        times = []
        for _ in range(5):
            start = perf_counter_ns()
            for _ in range(10):
                ttnn.execute_trace(device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            times.append((perf_counter_ns() - start) / 1e7)
        return median(times)
    finally:
        ttnn.release_trace(device, trace)


def run_layer(device, args, layer):
    prefix = f"layers.{layer}.mlp."
    with safe_open(args.weights / f"layer{layer:02d}.safetensors", framework="pt") as file:
        gu = file.get_tensor(prefix + "experts.gate_up_proj")
        dn = file.get_tensor(prefix + "experts.down_proj")
        rw = file.get_tensor(prefix + "gate.weight").float().T.contiguous()
        rb = file.get_tensor(prefix + "gate.e_score_correction_bias").float().flatten()

    rng = torch.Generator().manual_seed(19 + layer)
    cases = {}
    for tokens in args.tokens:
        x = torch.randn((1, 1, tokens, D), generator=rng)
        x = (x / x.square().mean(-1, keepdim=True).sqrt()).bfloat16()
        routing = route(x, rw, rb, tokens)
        cases[tokens] = (x, upload(device, x), routing)

    targets, fp32_targets = {}, {}
    if not args.skip_stock:
        stage, baseline = stock(device, gu, dn)
        for tokens, (_, xt, routing) in cases.items():
            emit(layer=layer, tokens=tokens, running="stock")
            scores = stage(routing, tokens)
            targets[tokens], elapsed = benchmark(device, lambda: baseline(xt, scores), args.repeats)
            _, end_to_end = benchmark(
                device,
                lambda: baseline(xt, stage(route(ttnn.to_torch(xt), rw, rb, tokens), tokens)),
                args.repeats,
                reference=targets[tokens],
            )
            emit(layer=layer, tokens=tokens, stock_ms=elapsed, stock_with_router_ms=end_to_end)
            fp32_targets[tokens] = ttnn.to_torch(baseline(xt, scores, fp32_epilogue=True))
        del stage, baseline
        ttnn.synchronize_device(device)

    handle = experts_stream.prepare(device, gu, dn)
    for tokens, (x, xt, routing) in cases.items():
        emit(layer=layer, tokens=tokens, running="packed", experts=routing[0].unique().numel())
        ws = experts_stream.prepare_prefill(device, tokens)
        experts_stream.stage_prefill_routing(routing, ws)
        actual, elapsed = benchmark(
            device,
            lambda: experts_stream.moe_prefill(xt, None, handle, workspace=ws),
            args.repeats,
        )
        _, end_to_end = benchmark(
            device,
            lambda: experts_stream.moe_prefill(xt, route(ttnn.to_torch(xt), rw, rb, tokens), handle, workspace=ws),
            args.repeats,
            reference=actual,
        )
        emit(layer=layer, tokens=tokens, packed_ms=elapsed, packed_with_router_ms=end_to_end)
        if tokens in targets:
            error = metrics(actual, targets[tokens])
            emit(layer=layer, tokens=tokens, packed_vs_stock=error)
            formula_error = metrics(actual, fp32_targets[tokens])
            emit(layer=layer, tokens=tokens, packed_vs_fp32_epilogue=formula_error)
            assert error["rel_l2"] < 0.01, error
            assert formula_error["equal_bits"] == formula_error["elements"], formula_error

        if args.trace:
            # Stress sparse routes and experts with more than one 8-row tile.
            # Only nine rows are valid and select the same eight experts.
            sparse_ids = torch.arange(7, -1, -1).expand(9, -1)
            sparse_gamma = torch.rand((9, K), generator=rng)
            sparse_gamma *= 8 / sparse_gamma.sum(-1, keepdim=True)
            sparse = (sparse_ids, sparse_gamma)
            partial = ttnn.to_torch(experts_stream.moe_prefill(xt, sparse, handle, workspace=ws))
            assert partial[:, :, :9].isfinite().all() and partial[:, :, :9].abs().max() > 0
            assert partial[:, :, 9:].count_nonzero() == 0
            replay_ms = trace_replay(device, xt, [sparse, routing], ws, handle)
            emit(layer=layer, tokens=tokens, packed_trace_ms=replay_ms, changing_routes_bitwise=True)
        ws = None
        if tokens == args.tokens[0]:
            x_l1 = upload(device, x, memory=ttnn.L1_MEMORY_CONFIG)
            one_call = ttnn.to_torch(experts_stream.moe_prefill(x_l1, routing, handle))
            assert torch.equal(one_call.view(torch.int16), actual.view(torch.int16))
            emit(layer=layer, tokens=tokens, automatic_workspace_and_l1_bitwise=True)
        if not args.skip_decode:
            decode, decode_ms = token_decode(device, x, routing, handle, args.repeats)
            emit(layer=layer, tokens=tokens, decode_ms=decode_ms, packed_vs_decode=metrics(actual, decode))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path(paths.hf_sd()))
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 20])
    parser.add_argument("--tokens", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-stock", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=16 << 20)
    try:
        device.enable_program_cache()
        for layer in args.layers:
            run_layer(device, args, layer)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
