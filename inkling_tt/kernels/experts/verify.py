"""Verify and time the batch-one expert stream on real Inkling layers."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch
import ttnn
from safetensors import safe_open

from inkling_tt import dense, paths, rowvm
from inkling_tt import experts_stream as stream

D, H, TOPK = 2304, 1152, 8
BF, FP, BFP8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
PAYLOAD = TOPK * (72 * 72 + 36 * 72) * 1088


def upload(device, x, dtype, layout=TILE, memory=L1):
    return ttnn.from_torch(x, device=device, dtype=dtype, layout=layout, memory_config=memory)


def stock(device, gate_up, down, fidelity):
    w_gu = upload(device, gate_up.transpose(1, 2)[None].contiguous(), BFP8, memory=DRAM)
    w_down = upload(device, down.transpose(1, 2)[None].contiguous(), BFP8, memory=DRAM)
    dummy = upload(device, torch.ones((1, 1, 1, 128)), BF, RM)
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    def pc(kw):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(9, 8),
            in0_block_w=kw,
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

    pc_gu, pc_down = pc(8), pc(6)

    def step(x, ids, gamma):
        common = dict(
            sparsity=dummy,
            indices=ids,
            is_input_b_sparse=True,
            memory_config=L1,
            dtype=BF,
            compute_kernel_config=ck,
        )
        gu = ttnn.sparse_matmul(x, w_gu, program_config=pc_gu, is_input_a_sparse=False, **common)
        h = rowvm.moe_hidden(ttnn.reshape(gu, (1, TOPK, 1, 2 * H)))
        d = ttnn.sparse_matmul(h, w_down, program_config=pc_down, is_input_a_sparse=True, **common)
        return rowvm.moe_combine(ttnn.reshape(d, (1, TOPK, 1, D)), gamma)

    return step


def reference(x, ids, weights, gate_up, down, *, round_matmuls=True):
    x = x.float().reshape(D)
    ids = ids.reshape(TOPK).long()
    weights = weights.reshape(TOPK)
    y = torch.zeros(D)
    for expert, gamma in zip(ids, weights):
        z = gate_up[expert].float() @ x
        if round_matmuls:
            z = z.bfloat16().float()
        g, u = z.chunk(2)
        h = ((4 * torch.tanh(g / 4)) * torch.sigmoid(g)) * (25 * torch.tanh(u / 25))
        if round_matmuls:
            h = h.bfloat16().float()
        d = down[expert].float() @ h
        if round_matmuls:
            d = d.bfloat16().float()
        y += d * gamma
    return y.reshape(1, 1, 1, D).bfloat16() if round_matmuls else y.reshape(1, 1, 1, D)


def compare(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    error = a - b
    return {
        "rel_l2": float(error.norm() / b.norm()),
        "max_abs": float(error.abs().max()),
        "equal": int((a == b).sum()),
    }


def show(**data):
    print(json.dumps(data), flush=True)


def mark(name):
    from tracy import signpost

    signpost(name)


def trace_timing(device, step, cases, name, steps, repeats, visits):
    def sequence():
        return [step(*cases[i % len(cases)][3]) for i in range(visits)]

    outputs = sequence()
    ttnn.synchronize_device(device)
    mark(f"{name}_capture")
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        outputs = sequence()
    finally:
        ttnn.end_trace_capture(device, tid, cq_id=0)
    try:
        ttnn.synchronize_device(device)
        mark(f"{name}_replay")
        samples = []
        for _ in range(repeats):
            start = perf_counter_ns()
            for _ in range(steps):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            samples.append((perf_counter_ns() - start) / (1000 * steps * visits))
        show(
            timing=name,
            trace_id=int(tid),
            visits=visits,
            median_us=median(samples),
            gbps=PAYLOAD / (median(samples) * 1000),
            samples_us=samples,
        )
        mark(f"{name}_end")
    finally:
        ttnn.release_trace(device, tid)
    return outputs


def layer(device, root, number, args):
    prefix = f"layers.{number}.mlp."
    with safe_open(root / f"layer{number:02d}.safetensors", framework="pt") as file:
        gate_up = file.get_tensor(prefix + "experts.gate_up_proj")
        down = file.get_tensor(prefix + "experts.down_proj")
        rw = file.get_tensor(prefix + "gate.weight").bfloat16()
        bias = file.get_tensor(prefix + "gate.e_score_correction_bias").float().reshape(1, 1, 1, 128)
    generator = torch.Generator().manual_seed(19 + number)
    xs = [torch.randn((1, 1, 1, D), generator=generator).bfloat16() for _ in range(args.cases + 1)]
    cases = []
    for i, x in enumerate(xs[:-1]):
        ids = torch.randperm(128, generator=generator)[:TOPK]
        if i == 1:
            ids = torch.cat((torch.tensor([127, 0]), ids[(ids != 0) & (ids != 127)][: TOPK - 2]))
        ids = ids.reshape(1, 1, 1, TOPK).to(torch.int16)
        gamma = torch.rand(ids.shape, generator=generator)
        gamma = 8 * gamma / gamma.sum()
        memory = DRAM if args.placement == "dram" or (args.placement == "mixed" and i == 1) else L1
        tensors = (
            upload(device, x, BF, memory=memory),
            upload(device, ids, ttnn.uint16, RM, memory=memory),
            upload(device, gamma, FP, memory=memory),
        )
        cases.append((x, ids, gamma, tensors))
    rw_tt = dense.prepare_weight(device, rw.T.contiguous(), BF)
    bias_tt = upload(device, bias, FP, memory=DRAM)
    x = xs[-1]
    xt = upload(device, x, BF, memory=DRAM if args.placement == "dram" else L1)
    ids_tt, gamma_tt = rowvm.router(dense.linear(xt, rw_tt), bias_tt)
    ids = ttnn.to_torch(ids_tt)
    gamma = ttnn.to_torch(gamma_tt)
    scores = (x.float() @ rw.float().T).sigmoid()
    expected_ids = (scores + bias).topk(TOPK).indices.sort(dim=-1).values
    assert torch.equal(ids.long().sort(dim=-1).values, expected_ids), (ids, expected_ids)
    expected_gamma = scores.gather(-1, ids.long())
    expected_gamma *= 8 / expected_gamma.sum()
    torch.testing.assert_close(gamma, expected_gamma, atol=2e-4, rtol=0)
    cases.append((x, ids, gamma, (xt, ids_tt, gamma_tt)))
    goldens = [reference(*case[:3], gate_up, down) for case in cases]
    fp32 = [reference(*case[:3], gate_up, down, round_matmuls=False) for case in cases]
    baseline = stock(device, gate_up, down, ttnn.MathFidelity.HiFi4)
    stock_values = [ttnn.to_torch(baseline(*case[3])) for case in cases]
    if args.steps:
        trace_timing(device, baseline, cases, f"L{number}_stock", args.steps, args.repeats, args.visits)
    del baseline
    ttnn.synchronize_device(device)

    handle = stream.prepare(device, gate_up, down)
    eager_hi4 = []
    for fidelity in (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2):
        impl = replace(handle, fidelity=fidelity)
        for i, case in enumerate(cases):
            actual = ttnn.to_torch(stream.moe_decode(*case[3], impl))
            stock_err = compare(actual, stock_values[i])
            boundary_err = compare(actual, goldens[i])
            fp32_err = compare(actual, fp32[i])
            stock_boundary_err = compare(stock_values[i], goldens[i])
            stock_fp32_err = compare(stock_values[i], fp32[i])
            if fidelity == ttnn.MathFidelity.HiFi4:
                eager_hi4.append(actual)
            show(
                layer=number,
                case="router" if i == len(cases) - 1 else i,
                fidelity=fidelity.name,
                stock=stock_err,
                hf_boundary=boundary_err,
                hf_fp32=fp32_err,
                stock_hf_boundary=stock_boundary_err,
                stock_hf_fp32=stock_fp32_err,
            )
            if fidelity == ttnn.MathFidelity.HiFi4:
                assert torch.equal(actual.view(torch.int16), stock_values[i].view(torch.int16)), stock_err
        if args.steps:
            trace_timing(
                device,
                lambda x, ids, gamma, impl=impl: stream.moe_decode(x, ids, gamma, impl),
                cases,
                f"L{number}_{fidelity.name}",
                args.steps,
                args.repeats,
                args.visits,
            )

    # Replay the exact same addresses after changing all three inputs.
    x_tt, ids_tt, gamma_tt = cases[0][3]

    def stage(case):
        for src, dst in zip(case[:3], (x_tt, ids_tt, gamma_tt)):
            host = ttnn.from_torch(src, dtype=dst.dtype, layout=dst.layout)
            ttnn.copy_host_to_device_tensor(host, dst)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        replay = stream.moe_decode(x_tt, ids_tt, gamma_tt, handle)
    finally:
        ttnn.end_trace_capture(device, tid, cq_id=0)
    try:
        for i in (-1, 0):
            stage(cases[i])
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            actual = ttnn.to_torch(replay)
            expected = eager_hi4[i]
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), compare(actual, expected)
    finally:
        ttnn.release_trace(device, tid)
    show(layer=number, changing_route_trace="bitwise equal to eager")

    def route_and_decode():
        ids, gamma = rowvm.router(dense.linear(xt, rw_tt), bias_tt)
        return stream.moe_decode(xt, ids, gamma, handle)

    return route_and_decode, eager_hi4[-1], handle.gate_up.buffer_address()


def verify_routed_trace(device, graphs, layers):
    assert len({address for _, _, address in graphs}) == len(graphs)
    for step, _, _ in graphs:
        step()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        outputs = [step() for step, _, _ in graphs]
    finally:
        ttnn.end_trace_capture(device, tid, cq_id=0)
    try:
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        for output, (_, expected, _) in zip(outputs, graphs):
            actual = ttnn.to_torch(output)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), compare(actual, expected)
    finally:
        ttnn.release_trace(device, tid)
    show(layers=layers, routed_trace="bitwise equal to eager")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path(paths.hf_sd()))
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 20])
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--visits", type=int, default=8)
    parser.add_argument("--placement", choices=("l1", "dram", "mixed"), default="mixed")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.cases, args.repeats, args.visits) < 1 or args.steps < 0:
        parser.error("cases, repeats and visits must be positive, and steps must be nonnegative")
    torch.set_num_threads(4)
    device = ttnn.open_device(device_id=0, trace_region_size=16 << 20)
    try:
        device.enable_program_cache()
        grid = device.compute_with_storage_grid_size()
        workers = device.get_optimal_dram_bank_to_logical_worker_assignment(
            ttnn.NOC.NOC_0,
            ttnn.MeshCoordinate(0, 0),
        )
        show(
            grid=[grid.x, grid.y],
            preferred_workers=[[workers[b].x, workers[b].y] for b in sorted(workers)],
            visit_bytes=PAYLOAD,
            placement=args.placement,
        )
        graphs = [layer(device, args.weights, number, args) for number in args.layers]
        verify_routed_trace(device, graphs, args.layers)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
