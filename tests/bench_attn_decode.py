"""Chip-local fused attention correctness and trace timings for Inkling-2-Mini."""

import argparse
import math
import statistics
import time

import torch
import ttnn

from inkling_tt import attn_decode, dense, rowvm
from inkling_tt.model import NEG, W


def relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def trace_calls(device, call, repeats=64):
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(repeats):
        call()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    return trace


def trace_us(device, trace, repeats=64):
    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(4):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) * 1e6 / (4 * repeats))
    return statistics.median(samples)


def run(device, length, *, full_test):
    global_cache = length == 2048
    packed_gate_col = 3488
    bf, rm, tile = ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

    def upload(t, dtype=bf, layout=tile, mem=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(t, device=device, dtype=dtype, layout=layout, memory_config=mem)

    def alloc(shape, mem=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), bf, tile, device, mem)

    ot = dense._l1_shards(device, 3072, 4)
    gt = dense._l1_shards(device, 6560, 4)
    q_h = torch.randn(1, 1, 24, 128).bfloat16()
    k_h = torch.randn(1, 8, length, 128).bfloat16()
    v_h = torch.randn_like(k_h)
    r_h = torch.zeros(1, 1, 24, 32, dtype=torch.bfloat16)
    r_h[..., :16] = torch.randn(24, 16)
    t_h = torch.zeros(W + 1, 32, dtype=torch.bfloat16)
    t_h[:W, :16] = torch.randn(W, 16) * 4
    s_h = torch.zeros(32, 32, dtype=torch.bfloat16)
    s_h[:24, 0] = torch.randn(24) * 12
    g_h = torch.randn(1, 1, 1, 6560).bfloat16()
    q, k, v, r, sink = map(upload, (q_h, k_h, v_h, r_h, s_h))
    # Stock SDPA requires DRAM for interleaved Q. The fused graph keeps
    # the three small attention-head inputs in interleaved L1.
    fused_q, fused_r, fused_sink = (upload(x, mem=ttnn.L1_MEMORY_CONFIG) for x in (q_h, r_h, s_h))
    table = upload(t_h, layout=rm)
    packed_table = attn_decode.prepare_table(table, global_cache=global_cache)
    gate = upload(g_h, mem=gt)
    pos = upload(torch.zeros(1, dtype=torch.int32), ttnn.int32, rm)
    y = alloc((1, 1, 1, 3072), ot)
    stock_y = alloc((1, 1, 1, 3072), ot)
    bias = alloc((1, 1, 24, length))

    ck = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
        q_chunk_size=32,
        k_chunk_size=128,
        exp_approx_mode=False,
    )

    def set_position(p):
        host = ttnn.from_torch(torch.tensor([p], dtype=torch.int32), dtype=ttnn.int32, layout=rm)
        ttnn.copy_host_to_device_tensor(host, pos)

    def stock():
        rowvm.relative_bias_from_position(
            r,
            pos,
            table,
            length,
            window=W,
            global_cache=global_cache,
            masked=NEG,
            out=bias,
        )
        out = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=bias,
            attention_sink=sink,
            scale=1 / math.sqrt(128),
            program_config=pc,
            compute_kernel_config=ck,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rowvm.gate_from_heads(out, gate, col_offset=packed_gate_col, out=stock_y)

    def fused():
        attn_decode.decode(
            fused_q,
            k,
            v,
            fused_r,
            fused_sink,
            packed_table,
            pos,
            gate,
            global_cache=global_cache,
            gate_col=packed_gate_col,
            out=y,
        )

    def reference(p):
        slots = torch.arange(length)
        d = p - slots
        ids = torch.where((d >= 0) & (d < W), d, W) if global_cache else d % W
        b = r_h.flatten(0, 2).float() @ t_h[ids].float().T
        b = torch.where(slots <= p, b, NEG).bfloat16().float()
        scores = q_h.reshape(24, 1, 128).float() @ k_h[0].repeat_interleave(3, 0).float().transpose(-1, -2)
        logits = torch.cat((scores[:, 0, :] + b, s_h[:24, :1].float()), dim=1) / math.sqrt(128)
        probs = logits.softmax(dim=-1)[:, :-1].bfloat16().float()
        out = probs[:, None, :] @ v_h[0].repeat_interleave(3, 0).float()
        return (out.reshape(1, 1, 1, -1).bfloat16().float() * g_h[..., packed_gate_col:].float().sigmoid()).bfloat16()

    set_position(length - 1)
    print(f"L={length} warming stock", flush=True)
    stock()
    ttnn.synchronize_device(device)
    print(f"L={length} warming fused", flush=True)
    fused()
    ttnn.synchronize_device(device)
    a, b = ttnn.to_torch(y), ttnn.to_torch(stock_y)
    print(
        f"L={length} eager relative error: stock {relative_error(a, b):.6g}, "
        f"reference {relative_error(a, reference(length - 1)):.6g}",
        flush=True,
    )
    if not full_test:
        return

    positions = (0, 31, 32, 127, 128, 733, 1023, 1024, 1500, 2047)
    stock_outputs = {}
    stock_us = {}
    timing_positions = (31, 733, length - 1, 2047)
    stock_trace = trace_calls(device, stock)
    for p in positions:
        set_position(p)
        ttnn.execute_trace(device, stock_trace, cq_id=0, blocking=True)
        stock_outputs[p] = ttnn.to_torch(stock_y)
        if p in timing_positions:
            stock_us[p] = trace_us(device, stock_trace)
    ttnn.release_trace(device, stock_trace)

    fused_trace = trace_calls(device, fused)
    for p in positions:
        set_position(p)
        ttnn.execute_trace(device, fused_trace, cq_id=0, blocking=True)
        a, b, ref = ttnn.to_torch(y), stock_outputs[p], reference(p)
        print(
            f"L={length} pos={p} bitwise={torch.equal(a, b)} stock={relative_error(a, b):.6g} "
            f"reference={relative_error(a, ref):.6g} stock/reference={relative_error(b, ref):.6g}",
            flush=True,
        )
        torch.testing.assert_close(a.float(), ref.float(), atol=0.012, rtol=0.025)
        if p in timing_positions:
            print(
                f"L={length} pos={p} trace us: fused={trace_us(device, fused_trace):.2f} stock={stock_us[p]:.2f}",
                flush=True,
            )
    ttnn.release_trace(device, fused_trace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--length", type=int, choices=(1024, 2048))
    args = parser.parse_args()
    torch.manual_seed(42)
    dev = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=100_000_000)
    try:
        dev.enable_program_cache()
        for length in (args.length,) if args.length else (1024, 2048):
            run(dev, length, full_test=not args.eager)
    finally:
        ttnn.close_device(dev)
