"""Inkling-2-Mini text decoder on Tenstorrent (TTNN), milestone 2: batch 1, device-resident and trace-capturable.

Differences from model.py (M1): logical row count 1; per-step positional data lives in persistent device tensors
written before each replay; routing (fp32-accumulated BF16 router, sigmoid, biased top-8, unbiased renormalised
gamma) runs on device and feeds indexed sparse matmuls; conv histories are 4-row matrices updated in place; the
attention sink seeds the softmax instead of being concatenated.
"""

import math
import os
from types import SimpleNamespace
from typing import NamedTuple

import torch
import ttnn

from inkling_tt import attn_decode, dense, experts_stream, rowvm
from inkling_tt.grid import worker_grid
from inkling_tt.model import (
    DR,
    EPS,
    GLOBAL_LAYERS,
    HD,
    HKV,
    HQ,
    LOG_ALPHA,
    LOG_FLOOR,
    LOOP0,
    LOOP1,
    NB,
    NEG,
    NPASS,
    QK_CLIP,
    REL_CLIP,
    TOPK,
    VOCAB,
    D,
    E,
    F,
    Precision,
    W,
    Weights,
    visit_slot,
)

BF, FP, RM, TILE = ttnn.bfloat16, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
MTP_TAP_VISITS = {(5, 0): 0, (11, 1): 1, (17, 2): 2}


class _DecodeBank(NamedTuple):
    """One shared prefix row and a separate contiguous tail for each loop pass."""

    tensor: ttnn.Tensor
    count: int
    tail_start: int

    def append(self):
        row = self.tail_start + self.count - 1 if self.count else 0
        return self._replace(count=self.count + 1), row


class Ctx:
    _mesh_visit_fusion = False

    def __init__(self, device, prec: Precision):
        self.dev = device
        grid = worker_grid(device)
        self.prec = prec
        single_stream = device.get_num_devices() == 1 and getattr(prec, "experts", "stock") == "stream"
        self.visit_fusion = (
            (self._mesh_visit_fusion or single_stream)
            and prec.dense_dtype == BF
            and os.environ.get("INK_VISIT_FUSION", "1") != "0"
        )
        self.message_fusion = self.visit_fusion and os.environ.get("INK_MESSAGE_FUSION", "1") != "0"
        self.position_fusion = self.visit_fusion and os.environ.get("INK_POSITION_FUSION", "1") != "0"
        self.layout_fusion = self.visit_fusion and os.environ.get("INK_LAYOUT_FUSION", "1") != "0"
        self.shard_fusion = self.layout_fusion and os.environ.get("INK_SHARD_FUSION", "1") != "0"
        self.fused_attn_decode = (
            self.position_fusion
            and self.shard_fusion
            and device.arch() == ttnn.Arch.BLACKHOLE
            and os.environ.get("INK_ATTN_DECODE", "1") == "1"
        )
        if self.fused_attn_decode:
            self.dev.enable_program_cache()
        self.group_gates = self.shard_fusion and os.environ.get("INK_GROUP_GATES", "1") != "0"
        self.join_attention_prep = self.layout_fusion and os.environ.get("INK_JOIN_ATTN_PREP", "1") != "0"
        self.paired_kv = single_stream and self.join_attention_prep and os.environ.get("INK_PAIR_KV", "1") != "0"
        self.fused_kv_prep = (
            self.paired_kv and grid.x * grid.y >= 129 and os.environ.get("INK_FUSED_KV_PREP", "1") == "1"
        )
        # Decode-only rowvm rows (constants, bank, residual scratch) as row-major DRAM pages: one read per stripe.
        self.row_major_state = (
            self.visit_fusion and device.get_num_devices() == 1 and os.environ.get("INK_RM_STATE", "1") != "0"
        )
        self.host_prefetch = os.environ.get("INK_HOST_PREFETCH", "1") != "0"
        self.message_down_inputs = os.environ.get("INK_MSG_DOWN_INPUTS", "1") != "0"
        self.fused_router = self.row_major_state and os.environ.get("INK_FUSED_ROUTER", "1") != "0"
        self.fused_message_down = (
            self.row_major_state and self.group_gates and os.environ.get("INK_FUSED_MSG_DOWN", "1") != "0"
        )
        self.fused_head_tail = (
            self.visit_fusion and device.get_num_devices() == 1 and os.environ.get("INK_HEAD_TAIL", "1") != "0"
        )
        # Greedy steps feed their argmax back on device (embedding row, position, tau).
        self.device_feedback = (
            self.row_major_state and self.fused_head_tail and os.environ.get("INK_DEVICE_FEEDBACK", "1") != "0"
        )
        mk = lambda fid: ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.ck_exact = mk(ttnn.MathFidelity.HiFi4)
        self.ck_dense = mk(prec.dense_fidelity)
        self.ck_expert = mk(prec.expert_fidelity)
        self.ck_sparse = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=prec.expert_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.kv_shard = ttnn.create_sharded_memory_config(
            shape=(32, HD),
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        pc = lambda cx, cy, kw: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
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
        self.sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            q_chunk_size=32,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self.pc_gu = pc(9, 8, 8)
        self.pc_dn = pc(9, 8, 6)

    def decode_row(self, t):
        """Row-major DRAM copy of a TILE constant for the fused decode path; the TILE original serves prefill.
        Interleaved L1 reserves a page in every worker bank, so per-layer rows stay in DRAM."""
        if t is None or not self.row_major_state:
            return t
        return ttnn.from_torch(ttnn.to_torch(t), dtype=t.dtype, layout=RM, device=self.dev, memory_config=DRAM)

    def decode_query(self, q):
        return q if q is None or not self.row_major_state else (q[0], q[1], self.decode_row(q[2]))

    def up(self, t, dtype=BF, layout=TILE, mem=DRAM):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=self.dev, memory_config=mem)

    def host(self, t, dtype, layout):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout)

    def read(self, t):
        return ttnn.to_torch(t)

    def row(self, v, dtype=FP):
        return self.up(v.reshape(1, 1, 1, -1).float(), dtype)

    def lin(self, x, w, ck=None, dtype=BF):
        return ttnn.linear(x, w, compute_kernel_config=ck or self.ck_dense, dtype=dtype)

    def mm(self, a, b, dtype=FP, transpose_b=False):
        return ttnn.matmul(a, b, transpose_b=transpose_b, compute_kernel_config=self.ck_exact, dtype=dtype)


def act(op, p=None):
    return (
        ttnn.UnaryWithParam(getattr(ttnn.UnaryOpType, op))
        if p is None
        else ttnn.UnaryWithParam(getattr(ttnn.UnaryOpType, op), p)
    )


def f32(x):
    return x if x.dtype == FP else ttnn.typecast(x, FP)


def bf16(x):
    return x if x.dtype == BF else ttnn.typecast(x, BF)


def _message_down_ready(rows, messages):
    """The fused controller input reads interleaved FP32 TILE rows with matching placements per side."""
    tensors = (*rows, *messages)
    if any(t.dtype != FP or t.layout != TILE or t.is_sharded() for t in tensors):
        return False
    return (
        len({t.memory_config().buffer_type for t in rows}) == 1
        and len({t.memory_config().buffer_type for t in messages}) == 1
    )


def _linear_to_shards(x, weight, *, activation=None, out=None):
    """Use the tuned dense program and keep its native output shards."""
    if x.memory_config() != weight.input_shards:
        x = (
            ttnn.to_memory_config(x, weight.input_shards)
            if activation is None
            else ttnn.interleaved_to_sharded(x, weight.input_shards, preallocated_output=activation)
        )
    return ttnn.linear(
        x,
        weight.tensor,
        program_config=weight.program,
        compute_kernel_config=weight.compute,
        memory_config=weight.output_shards,
        dtype=BF,
        optional_output_tensor=out,
    )


class Ops:
    def __init__(self, c: Ctx):
        self.c = c
        self.seg = {}
        shift = torch.zeros(4, 4)
        for r in range(1, 4):
            shift[r, r - 1] = 1.0
        e0 = torch.zeros(4, 1)
        e0[0, 0] = 1.0
        self.shift4 = c.up(shift.reshape(1, 1, 4, 4), FP)
        self.e0 = c.up(e0.reshape(1, 1, 4, 1), FP)
        self.ones14 = c.up(torch.ones(1, 1, 1, 4), FP)
        self.rowsel, self.rowmask = [], []
        for j in range(12):
            e = torch.zeros(1, 1, 32, 1)
            e[0, 0, j, 0] = 1.0
            self.rowsel.append(c.up(e, FP))
            mk = torch.full((1, 1, 32, 1), NEG)
            mk[0, 0, : j + 1, 0] = 0.0
            self.rowmask.append(c.up(mk, FP))
        self.bank0 = (c.up(torch.zeros(1, 1, 32, D), FP), 0)

    def segment(self, nheads, hd):
        key = (nheads, hd)
        if key not in self.seg:
            m = torch.zeros(nheads * hd, 32)
            for h in range(nheads):
                m[h * hd : (h + 1) * hd, h] = 1.0
            self.seg[key] = (
                self.c.up(m.reshape(1, 1, nheads * hd, 32), FP),
                self.c.up(m.T.reshape(1, 1, 32, nheads * hd), FP),
            )
        return self.seg[key]

    def rms(self, x, w=None, dtype=FP):
        if x.shape[-2] == 1 and x.shape[-3] == 1:
            return rowvm.rms_norm(x, w, dtype=dtype)
        y = ttnn.rms_norm(f32(x), epsilon=EPS, compute_kernel_config=self.c.ck_exact)
        if w is None:
            return y if dtype == FP else ttnn.typecast(y, dtype)
        return ttnn.multiply(y, w, dtype=dtype)

    def head_sumsq(self, x, nheads, hd):
        return self.c.mm(ttnn.multiply(x, x), self.segment(nheads, hd)[0])

    def head_bcast(self, v, nheads, hd):
        return self.c.mm(v, self.segment(nheads, hd)[1])

    def l2clip(self, x, nheads, hd, thr, scale=None):
        x = f32(x)
        if scale is not None:
            x = ttnn.multiply(x, scale)
        norm = ttnn.sqrt(self.head_sumsq(x, nheads, hd))
        fac = ttnn.minimum(ttnn.multiply(ttnn.reciprocal(ttnn.add(norm, EPS)), thr), 1.0)
        return ttnn.multiply(x, self.head_bcast(fac, nheads, hd))

    def head_rms(self, x, nheads, hd):
        x = f32(x)
        ms = ttnn.multiply(self.head_sumsq(x, nheads, hd), 1.0 / hd)
        return ttnn.multiply(x, self.head_bcast(ttnn.rsqrt(ttnn.add(ms, EPS)), nheads, hd))

    def hclip(self, x, thr, scale=None, out=FP):
        """Per-head L2 clip on [..., H, d] (reduction over the last dim)."""
        x = f32(x)
        if scale is not None:
            x = ttnn.multiply(x, scale)
        ss = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
        fac = [act("SQRT"), act("ADD_UNARY_SFPU", EPS), act("RECIP"), act("MUL_UNARY_SFPU", thr), act("MINIMUM", 1.0)]
        return ttnn.multiply(x, ss, input_tensor_b_activations=fac, dtype=out)

    def hrms(self, x, out=FP):
        x = f32(x)
        ss = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
        fac = [act("MUL_UNARY_SFPU", 1.0 / x.shape[-1]), act("ADD_UNARY_SFPU", EPS), act("RSQRT")]
        return ttnn.multiply(x, ss, input_tensor_b_activations=fac, dtype=out)

    def softcap(self, x, cap):
        return ttnn.multiply(ttnn.tanh(ttnn.multiply(x, 1.0 / cap)), cap)

    def bank_append(self, bank, v):
        """bank = (S [1,1,32,2304] fp32 normalized rows, J); returns a new bank with rms(v) as row J."""
        S, J = bank
        return (ttnn.add(S, self.c.mm(self.rowsel[J], self.rms(v))), J + 1)

    def mix(self, bank, partial, qmat, nq, weight=None, dtype=FP, *, out=None):
        """AttnRes depth mixing: softmax over the J bank rows plus rms(partial), one column per query.
        With weight [1,1,nq,D], each output is additionally RMS-normalised with it (and cast to dtype)."""
        if isinstance(bank, _DecodeBank):
            return list(
                rowvm.attnres_mix(
                    bank.tensor,
                    partial,
                    qmat[2],
                    bank.count,
                    bank_start=bank.tail_start,
                    shared_prefix=True,
                    weight=weight,
                    dtype=dtype,
                    out=out,
                )
            )
        S, J = bank
        return list(rowvm.attnres_mix(S, partial, qmat[2], J, weight=weight, dtype=dtype, out=out))
        S = ttnn.add(S, self.c.mm(self.rowsel[J], self.rms(partial)))
        if nq == 1:
            lg = ttnn.add(ttnn.sum(ttnn.multiply(S, qmat[1]), dim=-1, keepdim=True), self.rowmask[J])
        else:
            lg = ttnn.add(self.c.mm(S, qmat[0]), self.rowmask[J])
        p = ttnn.softmax(lg, dim=-2, numeric_stable=True, compute_kernel_config=self.c.ck_exact)
        out = ttnn.matmul(p, S, transpose_a=True, compute_kernel_config=self.c.ck_exact, dtype=FP)
        return [ttnn.slice(out, [0, 0, q, 0], [1, 1, q + 1, D]) for q in range(nq)]

    def conv_push(self, S, x):
        """S [1,1,4,C] fp32 rows (t, t-1, t-2, t-3) -> shifted with x as the new row 0, written in place."""
        new = ttnn.add(self.c.mm(self.shift4, S), self.c.mm(self.e0, f32(x)))
        ttnn.copy(new, S)
        return S

    def conv_apply(self, S, taps):
        return self.c.mm(self.ones14, ttnn.multiply(S, taps))


class Step:
    """Persistent per-step inputs (host writes them before each replay)."""

    def __init__(self, c: Ctx, lmax):
        self.c, self.lmax = c, lmax
        # Only rowvm reads the embedding row on the fused path; a row-major row avoids host tilization.
        self.x_layout = RM if c.row_major_state else TILE
        self.x = c.up(torch.zeros(1, 1, 1, D), BF, self.x_layout)
        self.pos_g = c.up(torch.zeros(1, dtype=torch.int32), ttnn.int32, RM)
        self.pos_l = c.up(torch.zeros(1, dtype=torch.int32), ttnn.int32, RM)
        self.idx_g = c.up(torch.zeros(1, lmax, dtype=torch.int32), ttnn.uint32, RM)
        self.idx_l = c.up(torch.zeros(1, W, dtype=torch.int32), ttnn.uint32, RM)
        self.mask_g = c.up(torch.zeros(1, 1, 1, lmax), FP)
        self.mask_l = c.up(torch.zeros(1, 1, 1, W), FP)
        self.tau = c.up(torch.ones(1, 1, 1, 1), FP)

    def _position_rows(self, pos):
        """Host position and tau tensors, built once per position."""
        cache = self.__dict__.setdefault("_positions", {})
        rows = cache.get(pos)
        if rows is None:
            tau = 1.0 + LOG_ALPHA * math.log(max((pos + 1) / LOG_FLOOR, 1.0))
            H = self.c.host
            rows = cache[pos] = (
                H(torch.tensor([pos], dtype=torch.int32), ttnn.int32, RM),
                H(torch.full((1, 1, 1, 1), tau), FP, TILE),
            )
        return rows

    def host_inputs(self, xrow, pos):
        H = self.c.host
        pos_row, tau_row = self._position_rows(pos)
        writes = [(H(xrow.reshape(1, 1, 1, D), BF, self.x_layout), self.x), (pos_row, self.pos_g), (tau_row, self.tau)]
        if not self.c.layout_fusion:
            writes.append((H(torch.tensor([pos % W], dtype=torch.int32), ttnn.int32, RM), self.pos_l))
        if self.c.position_fusion:
            return writes
        sg = torch.arange(self.lmax)
        dg = pos - sg
        idx_g = torch.where((dg >= 0) & (dg < W), dg, torch.full_like(dg, W))
        mask_g = torch.where(dg >= 0, 0.0, NEG)  # NEG*sqrt(HD) is still -inf-like
        sl = torch.arange(W)
        dl = (pos - sl) % W
        mask_l = torch.where(dl <= pos, 0.0, NEG)
        writes.extend(
            [
                (H(idx_g.reshape(1, -1).to(torch.int32), ttnn.uint32, RM), self.idx_g),
                (H(dl.reshape(1, -1).to(torch.int32), ttnn.uint32, RM), self.idx_l),
                (H(mask_g.reshape(1, 1, 1, -1), FP, TILE), self.mask_g),
                (H(mask_l.reshape(1, 1, 1, -1), FP, TILE), self.mask_l),
            ]
        )
        return writes

    def write(self, xrow, pos):
        self.device_state = None
        for h, d in self.host_inputs(xrow, pos):
            ttnn.copy_host_to_device_tensor(h, d)


class Visit:
    """Per-(layer, pass) caches: K/V and four conv histories."""

    def __init__(self, c: Ctx, is_global, lmax):
        self.klen = lmax if is_global else W
        z = torch.zeros(1, HKV, self.klen, HD)
        self.k = c.up(z)
        self.v = c.up(z)
        self.conv = [c.up(torch.zeros(1, 1, 4, n), FP) for n in (HKV * HD, HKV * HD, D, D)]


class Layer:
    def __init__(self, c: Ctx, ops: Ops, wts: Weights, i):
        self.c, self.ops, self.i = c, ops, i
        p = f"layers.{i}."
        g = lambda n: wts.get(p + n)
        self.is_global = i in GLOBAL_LAYERS
        self.is_loop = LOOP0 <= i < LOOP1
        self.block_boundary = i % NB == 0
        dd = c.prec.dense_dtype
        dev = c.dev
        KN = lambda w: w.T.contiguous()
        P = lambda w: dense.prepare_weight(dev, KN(w), dd)
        self._prepare_attention_weights(g)
        if self.is_global:
            self.norm_q = c.row(g("q_input_layernorm.weight"))
            self.norm_k = c.row(g("k_input_layernorm.weight"))
            self.norm_v = c.row(g("v_input_layernorm.weight"))
            self.norm_qkv = c.up(
                torch.stack([g(n + "_input_layernorm.weight").float() for n in "qkv"]).reshape(1, 1, 3, D), FP
            )
        else:
            self.norm_in = c.row(g("input_layernorm.weight"))
        self.norm_post = c.row(g("post_attention_layernorm.weight"))
        self.pa_taps = self._taps(g("post_attention_convolution.conv1d.weight"))
        self.pm_taps = self._taps(g("post_mlp_convolution.conv1d.weight"))
        self.q_attn = None
        if i > 0:
            qa = g("attention_input.query").float().reshape(-1, D)
            self.q_attn = qmat(c, qa)
        self.q_mlp = qmat(c, g("mlp_input.query").float().reshape(-1, D))
        if self.is_loop:
            self.gates = {}
            self.gate_down = {}
            for s in ("q", "k", "v", "mlp"):
                fw = g(f"{s}_message_gate.forget_gate_proj.weight").float()
                mw = g(f"{s}_message_gate.message_gate_proj.weight").float()
                self.gates[s] = (
                    P(g(f"{s}_message_gate.down_proj.weight")),
                    c.up(fw.T.reshape(1, 1, *fw.T.shape), FP),
                    c.up(mw.T.reshape(1, 1, *mw.T.shape), FP),
                )
                if c.fused_message_down:
                    # Tile-interleaved copy for the grouped decode projection; prefill keeps the sharded pack.
                    dw = KN(g(f"{s}_message_gate.down_proj.weight")).to(torch.bfloat16)
                    self.gate_down[s] = c.up(dw.reshape(1, 1, *dw.shape), BF)
        rw = g("mlp.gate.weight").float()
        assert torch.equal(rw, rw.bfloat16().float()), "router weights are expected to be BF16 values"
        self.router_w = dense.prepare_weight(dev, rw.T.contiguous().to(torch.bfloat16), ttnn.bfloat16)
        self.router_b = c.row(g("mlp.gate.e_score_correction_bias"))
        self.expert_stream = None
        if getattr(c.prec, "experts", "stock") == "stream":
            self.expert_stream = experts_stream.prepare(
                c.dev, g("mlp.experts.gate_up_proj").contiguous(), g("mlp.experts.down_proj").contiguous()
            )
        else:
            self.w_gu, self.w_dn = self._upload_experts(g("mlp.experts.gate_up_proj"), g("mlp.experts.down_proj"))
        R = c.decode_row
        self.dec = SimpleNamespace(
            q_attn=c.decode_query(self.q_attn),
            q_mlp=c.decode_query(self.q_mlp),
            norm_in=R(getattr(self, "norm_in", None)),
            norm_post=R(self.norm_post),
            norm_qkv=R(getattr(self, "norm_qkv", None)),
            norm_q=R(getattr(self, "norm_q", None)),
            norm_k=R(getattr(self, "norm_k", None)),
            norm_v=R(getattr(self, "norm_v", None)),
            pa_taps=R(self.pa_taps),
            pm_taps=R(self.pm_taps),
            k_taps=R(self.k_taps),
            v_taps=R(self.v_taps),
            router_b=R(self.router_b),
        )

    def _prepare_attention_weights(self, g):
        c, a = self.c, "self_attn."
        dev, dd = c.dev, c.prec.dense_dtype
        KN = lambda w: w.T.contiguous()
        S_ = lambda w, sc: (w.float() * sc.float().reshape(-1, 1)).to(torch.bfloat16)
        P = lambda w: dense.prepare_weight(dev, KN(w), dd)
        self.dq = dense.prepare_fused_weight(
            dev,
            (
                KN(S_(g(a + "q_proj.linear.weight"), g(a + "q_proj.norm.scale"))),
                KN(S_(g(a + "relative_position_bias.linear.weight"), g(a + "relative_position_bias.norm.scale"))),
                KN((g(a + "sink_proj.weight").float() * math.sqrt(HD)).to(torch.bfloat16)),
                KN(g(a + "o_proj.gate_proj.weight")),
            ),
            dd,
        )
        table = torch.zeros(W + 1, 32)
        table[:W, :DR] = g(a + "relative_position_bias.proj").float().T * math.sqrt(HD)
        self.rel_table = c.up(table, BF, RM)
        self.wo = P(g(a + "o_proj.linear.weight"))
        wk, wv = (KN(g(a + s + "_proj.linear.weight")) for s in "kv")
        self.wk, self.wv = (dense.prepare_weight(dev, w, dd) for w in (wk, wv))
        if c.paired_kv:
            self.decode_kv = dense.prepare_decode_kv(dev, wk, wv)
        if self.is_global:
            self.k_taps = self._dyn_taps(g(a + "k_proj.sconv.static_weight"))
            self.v_taps = self._dyn_taps(g(a + "v_proj.sconv.static_weight"))
            self.k_wproj = P(g(a + "k_proj.sconv.weight_proj.weight"))
            self.v_wproj = P(g(a + "v_proj.sconv.weight_proj.weight"))
            lagmask = torch.zeros(4, 32)
            sel = torch.zeros(32, HKV * HD)
            for h in range(HKV):
                for lag in range(4):
                    lagmask[lag, h * 4 + lag] = 1.0
                    sel[h * 4 + lag, h * HD : (h + 1) * HD] = 1.0 / 3.0
            self.lagmask = c.up(lagmask.reshape(1, 1, 4, 32), FP)
            self.ones41 = c.up(torch.ones(1, 1, 4, 1), FP)
            self.dyn_sel = c.up(sel.reshape(1, 1, 32, HKV * HD), FP)
        else:
            self.k_taps = self._taps(g(a + "k_proj.sconv.conv1d.weight"))
            self.v_taps = self._taps(g(a + "v_proj.sconv.conv1d.weight"))

    def _upload_experts(self, gu, dn):
        c, i = self.c, self.i
        ed = c.prec.expert_dtype(i) if callable(c.prec.expert_dtype) else c.prec.expert_dtype
        return (
            c.up(gu.transpose(1, 2).reshape(1, E, D, 2 * F), ed),
            c.up(dn.transpose(1, 2).reshape(1, E, F, D), ed),
        )

    def _taps(self, w):
        w = w.float().reshape(-1, 4)
        t = torch.stack([w[:, 3] + 1.0, w[:, 2], w[:, 1], w[:, 0]]) / math.sqrt(5.0)
        return self.c.up(t.reshape(1, 1, 4, -1), FP)

    def _dyn_taps(self, static):
        t = static.float().clone()
        t[0] += 1.0
        return self.c.up((t / 3.0).reshape(1, 1, 4, -1), FP)

    def sconv(self, x, taps, S, dtype=BF):
        return rowvm.depthwise_conv(S, x, taps, dtype=dtype)

    def dsconv(self, x, xin, taps, wproj, S, dtype=BF):
        return rowvm.depthwise_conv(S, x, taps, dyn=dense.linear(xin, wproj), dtype=dtype)

    def gate_msg(self, s, u, m, un=None, weight=None, *, out=None, controller_out=None):
        dn, wf, wg = self.gates[s]
        ops, c = self.ops, self.c
        if u.shape[-2] == 1:
            project = c.visit_fusion and c.message_fusion and weight is not None
            if c.fused_message_down and project:
                if c.message_down_inputs and un is None and _message_down_ready((u,), (m,)):
                    h = rowvm.message_down([(u, m, self.gate_down[s])], out=controller_out)[0]
                else:
                    hin = rowvm.message_input(u, m, normalized_u=un, memory_config=L1)
                    h = rowvm.matvec_group([(hin, self.gate_down[s])])[0]
                return rowvm.message_project(u, m, h, wf, wg, weight, out=out)
            if c.shard_fusion and project:
                h = _linear_to_shards(
                    rowvm.message_input(u, m, normalized_u=un, memory_config=dn.input_shards),
                    dn,
                )
                return rowvm.message_project(u, m, h, wf, wg, weight, out=out)
            h = dense.linear(rowvm.message_input(u, m, normalized_u=un), dn)
            if project:
                return rowvm.message_project(u, m, h, wf, wg, weight, out=out)
            if weight is None:
                return rowvm.message_output(u, m, c.mm(h, wf), c.mm(h, wg))
            return rowvm.message_output(u, m, c.mm(h, wf), c.mm(h, wg), weight=weight, dtype=BF, out=out)
        u, m = f32(u), f32(m)
        un = ops.rms(u) if un is None else un
        hin = bf16(ttnn.concat([un, ops.rms(m)], dim=-1))
        h = dense.linear(hin, dn) if hin.shape[-2] == 1 else f32(self.ops.prefill_lin(hin, dn))
        a1 = ttnn.multiply(u, c.mm(h, wf), input_tensor_b_activations=[act("TANH"), act("RSUB", 1.0)])
        return ttnn.add(a1, ttnn.multiply(m, c.mm(h, wg), input_tensor_b_activations=[act("TANH")]))

    def _gate_attention(self, rows, messages, norms, *, shared=False, scratch=None):
        c = self.c
        if c.shard_fusion and c.group_gates and c.message_fusion:
            gates = [self.gates[s] for s in "qkv"]
            kv_mem = L1 if c.paired_kv else None
            output_layouts = (self.dq.weight.input_shards, kv_mem, kv_mem) if c.join_attention_prep else (None,) * 3
            if c.fused_message_down:
                downs = [self.gate_down[s] for s in "qkv"]
                if c.message_down_inputs and _message_down_ready(rows, messages):
                    hs = rowvm.message_down(
                        zip(rows, messages, downs), out=None if scratch is None else scratch.qkv_down
                    )
                else:
                    controller_inputs = rowvm.qkv_message_inputs((u, m, L1) for u, m in zip(rows, messages))
                    hs = rowvm.matvec_group(zip(controller_inputs, downs))
                return rowvm.qkv_message_projections(
                    (
                        ((u, m, h, wf, wg, norm), mem)
                        for u, m, h, (dn, wf, wg), norm, mem in zip(rows, messages, hs, gates, norms, output_layouts)
                    ),
                    out=None if scratch is None else scratch.qkv_blend,
                )
            controller_inputs = rowvm.qkv_message_inputs(
                (u, m, gate[0].input_shards) for u, m, gate in zip(rows, messages, gates)
            )
            return rowvm.qkv_message_projections(
                ((u, m, _linear_to_shards(hin, dn), wf, wg, norm), mem)
                for u, m, hin, (dn, wf, wg), norm, mem in zip(
                    rows, messages, controller_inputs, gates, norms, output_layouts
                )
            )
        hn = self.ops.rms(rows[0]) if shared else None
        return tuple(self.gate_msg(s, u, m, hn, weight=norm) for s, u, m, norm in zip("qkv", rows, messages, norms))

    def attention(self, xq, xk, xv, vs: Visit, st: Step, *, staging=None, scratch=None):
        c = self.c
        if scratch is not None:
            qrsg = _linear_to_shards(xq, self.dq.weight, activation=scratch.sharded_q, out=scratch.qrsg)
        else:
            qrsg = _linear_to_shards(xq, self.dq.weight) if c.shard_fusion else dense.linear(xq, self.dq.weight)
        r_col = HQ * HD
        sink_col = r_col + HQ * DR
        gate_col = sink_col + 32
        tau = st.tau if self.is_global else None
        q, rp, sinkt, k_shard, v_shard, gated = c.attention_glue
        fused_decode = c.fused_attn_decode and (vs.klen == W or (vs.klen % 32 == 0 and vs.klen <= 32768))
        if fused_decode:
            # The eager graph in capture() allocates this scratch before tracing.
            if not hasattr(c, "_fused_attention_heads"):
                c._fused_attention_heads = tuple(
                    ttnn.allocate_tensor_on_device(t.shape, BF, TILE, c.dev, L1) for t in (q, rp, sinkt)
                )
            q, rp, sinkt = c._fused_attention_heads
        if c.layout_fusion and not c.join_attention_prep:
            rowvm.attention_heads(qrsg, q, rp, sinkt, q_threshold=QK_CLIP, r_threshold=REL_CLIP, tau=tau)
        elif not c.layout_fusion:
            rowvm.head_norm_from_row(qrsg, HQ, HD, threshold=QK_CLIP, tau=tau, out=q)
            rowvm.head_norm_from_row(
                qrsg,
                HQ,
                DR,
                col_offset=r_col,
                threshold=REL_CLIP,
                tau=tau,
                out_shape=(1, 1, HQ, 32),
                out=rp,
            )
            rowvm.sink_tile(qrsg, col_offset=sink_col, out=sinkt)

        def project(x, weight, dest):
            return ttnn.linear(
                x,
                weight.tensor,
                program_config=weight.program,
                compute_kernel_config=weight.compute,
                memory_config=dest.memory_config(),
                optional_output_tensor=dest,
            )

        kv_projection = None
        if scratch is None and c.paired_kv and xk.memory_config() == xv.memory_config():
            kr, vr = c.kv_rows
            if c.fused_kv_prep:
                kv_projection = (xk, xv, self.decode_kv)
            else:
                rowvm.project_kv(xk, xv, self.decode_kv, kr, vr)
        elif scratch is None:
            kr, vr = dense.linear(xk, self.wk), dense.linear(xv, self.wv)
        else:
            kr, vr = project(xk, self.wk, scratch.k), project(xv, self.wv, scratch.v)
        if c.layout_fusion:
            if staging is None:
                caches, cache_pos, wrap = (vs.k, vs.v), st.pos_g, not self.is_global
            else:
                staged_k, staged_v, _, cache_pos = staging
                caches, wrap = (staged_k, staged_v), False
            k_dyn = v_dyn = None
            if self.is_global:
                if scratch is None:
                    k_dyn, v_dyn = dense.linear(xk, self.k_wproj), dense.linear(xv, self.v_wproj)
                else:
                    k_dyn = project(xk, self.k_wproj, scratch.k_dyn)
                    v_dyn = project(xv, self.v_wproj, scratch.v_dyn)
            if c.join_attention_prep:
                rowvm.prepare_attention(
                    qrsg,
                    q,
                    rp,
                    sinkt,
                    kr,
                    vr,
                    caches,
                    vs.conv[:2],
                    (self.dec.k_taps, self.dec.v_taps),
                    cache_pos,
                    q_threshold=QK_CLIP,
                    r_threshold=REL_CLIP,
                    tau=tau,
                    k_dyn=k_dyn,
                    v_dyn=v_dyn,
                    wrap=wrap,
                    kv_projection=kv_projection,
                )
            else:
                rowvm.update_kv(
                    kr,
                    vr,
                    *caches,
                    vs.conv[0],
                    vs.conv[1],
                    self.dec.k_taps,
                    self.dec.v_taps,
                    cache_pos,
                    threshold=QK_CLIP,
                    k_dyn=k_dyn,
                    v_dyn=v_dyn,
                    wrap=wrap,
                )
        else:
            if self.is_global:
                k = self.dsconv(kr, xk, self.k_taps, self.k_wproj, vs.conv[0])
                v = self.dsconv(vr, xv, self.v_taps, self.v_wproj, vs.conv[1])
            else:
                k = self.sconv(kr, self.k_taps, vs.conv[0])
                v = self.sconv(vr, self.v_taps, vs.conv[1])
            rowvm.kv_norm_to_shard(k, threshold=QK_CLIP, memory_config=c.kv_shard, out=k_shard)
            rowvm.kv_norm_to_shard(v, memory_config=c.kv_shard, out=v_shard)
            pos = st.pos_g if self.is_global else st.pos_l
            for cache, t in ((vs.k, k_shard), (vs.v, v_shard)):
                ttnn.experimental.paged_update_cache(cache, t, update_idxs_tensor=pos)

        if fused_decode:
            # capture() runs this branch eagerly before opening the trace.
            if not hasattr(self, "_decode_rel_table"):
                self._decode_rel_table = attn_decode.prepare_table(self.rel_table, global_cache=self.is_global)
            attn_decode.decode(
                q,
                vs.k,
                vs.v,
                rp,
                sinkt,
                self._decode_rel_table,
                st.pos_g,
                qrsg,
                global_cache=self.is_global,
                gate_col=gate_col,
                out=c.gated_shards,
                staged=None if staging is None else staging[:3],
            )
            return _linear_to_shards(c.gated_shards, self.wo, out=None if scratch is None else scratch.attended)
        if c.position_fusion:
            bias = rowvm.relative_bias_from_position(
                rp,
                st.pos_g,
                self.rel_table,
                vs.klen,
                window=W,
                global_cache=self.is_global,
                masked=NEG,
                out=c.attention_bias[vs.klen],
            )
        else:
            idx, mask = (st.idx_g, st.mask_g) if self.is_global else (st.idx_l, st.mask_l)
            bias = rowvm.relative_bias(rp, idx, self.rel_table, mask, out=c.attention_bias[vs.klen])
        out = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            vs.k,
            vs.v,
            is_causal=False,
            attn_mask=bias,
            attention_sink=sinkt,
            scale=1.0 / math.sqrt(HD),
            program_config=c.sdpa_pc,
            compute_kernel_config=c.ck_exact,
            memory_config=DRAM,
        )
        if c.shard_fusion:
            rowvm.gate_from_heads(out, qrsg, col_offset=gate_col, out=c.gated_shards)
            return _linear_to_shards(c.gated_shards, self.wo)
        rowvm.gate_from_heads(out, qrsg, col_offset=gate_col, out=gated)
        return dense.linear(gated, self.wo)

    def moe(self, xm):
        c = self.c
        if c.fused_router and xm.dtype == BF:
            ids16, gamma = rowvm.fused_router(xm, self.router_w.tensor, self.dec.router_b)
        else:
            ids16, gamma = rowvm.router(dense.linear(xm, self.router_w), self.dec.router_b)
        if self.expert_stream is not None:
            return experts_stream.moe_decode(xm, ids16, gamma, self.expert_stream)
        common = dict(
            sparsity=self.ops.dummy,
            indices=ids16,
            is_input_b_sparse=True,
            memory_config=L1,
            dtype=BF,
            compute_kernel_config=c.ck_sparse,
        )
        gu = ttnn.sparse_matmul(xm, self.w_gu, program_config=c.pc_gu, is_input_a_sparse=False, **common)
        h = rowvm.moe_hidden(ttnn.reshape(gu, (1, TOPK, 1, 2 * F)))
        d = ttnn.sparse_matmul(h, self.w_dn, program_config=c.pc_dn, is_input_a_sparse=True, **common)
        return rowvm.moe_combine(ttnn.reshape(d, (1, TOPK, 1, D)), gamma)

    def _visit_moe(self, x, visit):
        return self.moe(x)

    def forward(self, x, bank, vs, st, messages=None):
        return self._forward(x, bank, vs, st, messages)[0]

    def _forward(
        self,
        x,
        bank,
        vs,
        st,
        messages=None,
        *,
        attention_input=None,
        next_mix=None,
        staging=None,
        defer_experts=False,
        scratch=None,
    ):
        ops, d = self.ops, self.dec
        block_residual = x
        fp_inputs = None if scratch is None else scratch.inputs_fp
        bf_inputs = None if scratch is None else scratch.inputs_bf

        def mix_attention(weight=None, dtype=FP, nq=1):
            buffers = fp_inputs if dtype == FP else bf_inputs
            return ops.mix(
                bank, x, d.q_attn, nq, weight=weight, dtype=dtype, out=None if buffers is None else buffers[:nq]
            )

        if self.is_global:
            if d.q_attn is None:
                xq = xk = xv = f32(x)
            elif messages is None:
                xq, xk, xv = mix_attention(d.norm_qkv, BF, 3)
            else:
                xq, xk, xv = mix_attention(nq=3)
            if messages is not None:
                xq, xk, xv = self._gate_attention(
                    (xq, xk, xv), messages, (d.norm_q, d.norm_k, d.norm_v), scratch=scratch
                )
        else:
            if d.q_attn is None:
                xq = xk = xv = (
                    ops.rms(x, d.norm_in, BF)
                    if scratch is None
                    else rowvm.rms_norm(x, d.norm_in, dtype=BF, out=bf_inputs[0])
                )
            elif messages is None:
                xq = xk = xv = attention_input
                if xq is None:
                    xq = xk = xv = mix_attention(d.norm_in, BF)[0]
            else:
                h = attention_input if attention_input is not None else mix_attention()[0]
                xq, xk, xv = self._gate_attention((h,) * 3, messages, (d.norm_in,) * 3, shared=True, scratch=scratch)
        a = self.attention(xq, xk, xv, vs, st, staging=staging, scratch=scratch)
        if self.block_boundary and not isinstance(bank, _DecodeBank):
            bank = ops.bank_append(bank, block_residual)
        if self.c.visit_fusion:
            z_bf, z_fp, xm = self.c.visit_scratch if scratch is None else scratch.visit
            z = z_bf if self.block_boundary else z_fp
            residual = None if self.block_boundary else block_residual
            append = None
            if isinstance(bank, _DecodeBank):
                S, J, start = bank
                if self.block_boundary:
                    bank, row = bank.append()
                    append = (block_residual, row)
            else:
                S, J = bank
                start = 0
            mixed = rowvm.attnres_mix(
                S,
                a,
                d.q_mlp[2],
                J,
                weight=d.norm_post if messages is None else None,
                dtype=BF if messages is None else FP,
                out=(xm,) if messages is None else None if scratch is None else (scratch.mixed_fp,),
                memory_config=L1 if messages is None else DRAM,
                convolution=(vs.conv[2], d.pa_taps, residual, z),
                bank_start=start,
                shared_prefix=bool(start),
                append=append,
            )[0]
            if messages is not None:
                mixed = self.gate_msg(
                    "mlp",
                    mixed,
                    messages[3],
                    weight=d.norm_post,
                    out=xm,
                    controller_out=None if scratch is None else scratch.mlp_down,
                )
            if defer_experts:
                return mixed, bank, z
            branch = self._visit_moe(mixed, vs)
            return self._finish_visit(branch, bank, z, vs.conv[3], next_mix)

        a = self.sconv(a, self.pa_taps, vs.conv[2])
        z = a if self.block_boundary else rowvm.add(block_residual, a, dtype=FP)
        if messages is None:
            xm = ops.mix(bank, z, self.q_mlp, 1, weight=self.norm_post, dtype=BF)[0]
        else:
            xm = self.gate_msg("mlp", ops.mix(bank, z, self.q_mlp, 1)[0], messages[3], weight=self.norm_post)
        y = self.sconv(self._visit_moe(xm, vs), self.pm_taps, vs.conv[3])
        return rowvm.add(z if z.dtype == FP else f32(z), f32(y), dtype=BF), bank, None

    def _finish_visit(self, branch, bank, residual, history, next_mix, *, out=None, mixed_out=None):
        taps = self.dec.pm_taps
        if next_mix is None:
            return rowvm.conv_residual(history, branch, taps, residual, out=out), bank, None
        query, weight = next_mix
        rm = self.c.row_major_state
        y = out
        if y is None:
            y = ttnn.allocate_tensor_on_device(
                ttnn.Shape(residual.shape), BF, RM if rm else TILE, self.c.dev, L1 if rm else DRAM
            )
        mix = rowvm.attnres_mix(
            bank.tensor,
            branch,
            query,
            bank.count,
            bank_start=bank.tail_start,
            shared_prefix=True,
            weight=weight,
            dtype=BF if weight is not None else FP,
            out=(mixed_out,) if mixed_out is not None else None,
            convolution=(history, taps, residual, y),
            round_residual=True,
            memory_config=L1 if self.c.paired_kv and weight is not None else DRAM,
        )[0]
        return y, bank, mix


def init_attention_glue(c, lmax):
    """Persistent attention scratch (allocated before trace capture)."""

    def alloc(shape, mem=DRAM):
        return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), BF, TILE, c.dev, mem)

    c.attention_glue = (
        alloc((1, 1, HQ, HD)),
        alloc((1, 1, HQ, 32)),
        alloc((32, 32)),
        alloc((1, 1, HKV, HD), c.kv_shard),
        alloc((1, 1, HKV, HD), c.kv_shard),
        alloc((1, 1, 1, HQ * HD)),
    )
    c.attention_bias = {width: alloc((1, 1, HQ, width)) for width in {W, lmax}}


def qmat(c, q):
    """(matrix [2304,32] with the queries as columns, row [1,1,1,2304] of query 0), pre-scaled by 1/sqrt(2304)."""
    m = torch.zeros(D, 32)
    m[:, : q.shape[0]] = q.T / math.sqrt(D)
    return (
        c.up(m.reshape(1, 1, D, 32), FP),
        c.up((q[0] / math.sqrt(D)).reshape(1, 1, 1, D), FP),
        c.up((q / math.sqrt(D)).reshape(1, 1, q.shape[0], D), FP),
    )


class InklingTT2:
    _ctx_type = Ctx
    _layer_type = Layer
    _visit_type = Visit

    def __init__(self, device, wpath, prec: Precision, lmax=2048, layers=None, log=print, *, capture_taps=False):
        n = 24 if layers is None else layers
        if capture_taps and n != 24:
            raise ValueError("MTP feature capture requires all 24 layers")
        self.c = self._ctx_type(device, prec)
        self.ops = Ops(self.c)
        self.ops.dummy = self.c.up(torch.zeros(1, 1, 1, E), BF, RM, L1)
        self.wts = Weights(wpath)
        g = self.wts.get
        self.embed = g("embed_tokens.weight")
        self.embed_norm = g("embed_tokens.embed_norm.weight").float()
        self.lmax = lmax
        self.layers = []
        for i in range(n):
            self.layers.append(self._layer_type(self.c, self.ops, self.wts, i))
            log(f"layer {i} loaded")
        self.final_q = qmat(self.c, g("final_input.query").float().reshape(-1, D))
        mq = g("message_input.query").float()
        self.msg_q = [qmat(self.c, mq[j]) for j in range(LOOP1 - LOOP0)]
        self.norm = self.c.row(g("norm.weight"))
        self.dec_final_q, self.dec_norm = self.c.decode_query(self.final_q), self.c.decode_row(self.norm)
        self.dec_msg_q = [self.c.decode_query(q) for q in self.msg_q]
        self._prepare_head(g("lm_head.weight"))
        self.st = Step(self.c, lmax)
        if self.c.device_feedback and hasattr(self, "head_logits"):
            self._prepare_feedback()
        self.next_id = None
        self.device_argmax = getattr(prec, "device_argmax", False)
        self.visits = {}
        for i in range(n):
            for p in range(NPASS if LOOP0 <= i < LOOP1 else 1):
                self.visits[visit_slot(i, p)] = self._visit_type(self.c, i in GLOBAL_LAYERS, lmax)
        self._init_attention_glue()
        if self.c.paired_kv:
            self.c.kv_rows = tuple(
                ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, HKV * HD)), BF, TILE, device, L1) for _ in range(2)
            )
        if self.c.visit_fusion:
            rm = self.c.row_major_state
            self.c.visit_scratch = tuple(
                ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, D)), dt, layout, device, mem)
                for dt, layout, mem in (
                    (BF, RM if rm else TILE, L1 if rm else DRAM),
                    (FP, RM if rm else TILE, L1 if rm else DRAM),
                    (BF, TILE, L1),
                )
            )
            self.decode_bank = (
                ttnn.from_torch(torch.zeros(1, 1, 32, D), dtype=FP, layout=RM, device=device, memory_config=L1)
                if rm
                else self.c.up(torch.zeros(1, 1, 32, D), FP)
            )
        self.trace_id = None
        self.logits = None
        self.taps = (
            tuple(ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, D)), BF, TILE, device, DRAM) for _ in range(4))
            if capture_taps
            else None
        )
        self._shift_cache = {}

    def _prepare_head(self, head):
        c = self.c
        head = torch.cat([head, torch.zeros(201024 - head.shape[0], D, dtype=head.dtype)], 0)
        self.head = dense.prepare_weight(
            c.dev, head.T.contiguous(), getattr(c.prec, "head_dtype", None) or c.prec.dense_dtype
        )
        vm = torch.zeros(1, 1, 1, 201024)
        vm[..., VOCAB:] = -1e4
        self.vocab_mask = c.up(vm, BF)
        if getattr(c, "fused_head_tail", False) and self.head.output_shards is not None:
            # Persistent outputs of the fused head tail, allocated before capture.
            self.head_logits = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, 201024)), BF, RM, c.dev, DRAM)
            self.head_index = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, 1)), ttnn.uint32, RM, c.dev, L1)

    def _init_attention_glue(self):
        init_attention_glue(self.c, self.lmax)
        if self.c.shard_fusion and self.layers:
            self.c.gated_shards = ttnn.allocate_tensor_on_device(
                ttnn.Shape((1, 1, 1, HQ * HD)),
                BF,
                TILE,
                self.c.dev,
                self.layers[0].wo.input_shards,
            )

    def _embedding_rows(self):
        """All normalized BF16 embedding rows, equal to embed_row for every token."""
        rows = torch.empty(self.embed.shape, dtype=torch.bfloat16)
        for a in range(0, rows.shape[0], 8192):
            e = self.embed[a : a + 8192].float()
            rows[a : a + 8192] = (e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + EPS) * self.embed_norm).to(
                torch.bfloat16
            )
        sample = torch.randint(0, rows.shape[0], (512,)).tolist() + [0, rows.shape[0] - 1]
        if any(not torch.equal(rows[t], self.embed_row(t)) for t in sample):
            for t in range(rows.shape[0]):
                rows[t] = self.embed_row(t)
        return rows

    def _prepare_feedback(self):
        c = self.c
        self.embed_table = ttnn.from_torch(
            self._embedding_rows(), dtype=BF, layout=RM, device=c.dev, memory_config=DRAM
        )
        tau = torch.tensor(
            [1.0 + LOG_ALPHA * math.log(max((p + 1) / LOG_FLOOR, 1.0)) for p in range(self.lmax)], dtype=torch.float32
        )
        self.tau_table = ttnn.from_torch(tau.reshape(1, -1), dtype=FP, layout=RM, device=c.dev, memory_config=DRAM)
        self.id_history = ttnn.from_torch(
            torch.zeros(1, self.lmax, dtype=torch.int32), dtype=ttnn.uint32, layout=RM, device=c.dev, memory_config=DRAM
        )
        self.st.device_state = None

    def embed_row(self, tok):
        e = self.embed[tok].float()
        return (e * torch.rsqrt(e.pow(2).mean() + EPS) * self.embed_norm).to(torch.bfloat16)

    def graph(self):
        """The whole decode step on device; reads Step tensors, returns BF16 logits [1,1,1,Vpad]."""
        ops, L = self.ops, self.layers
        if self.c.visit_fusion:
            return self._readout(self._fused_visits())
        x, bank = self.st.x, ops.bank0

        def run(layer, x, bank, p, messages=None):
            if self.taps is not None and (tap := MTP_TAP_VISITS.get((layer.i, p))) is not None:
                ttnn.copy(x, self.taps[tap])
            visit = self.visits[visit_slot(layer.i, p)]
            y = layer.forward(x, bank, visit, self.st, messages)
            if layer.block_boundary:
                bank = ops.bank_append(bank, x)
            return y, bank

        for layer in L[:LOOP0]:
            x, bank = run(layer, x, bank, 0)
        x_pre, bank_pre = x, bank
        npre = bank_pre[1]
        for layer in L[LOOP0:LOOP1]:
            x, bank = run(layer, x, bank, 0)
        for p in range(1, NPASS):
            prev_bank, prev_x = self.slice_bank(bank, npre + 1), x
            x, bank = x_pre, bank_pre
            for j, layer in enumerate(L[LOOP0:LOOP1]):
                x, bank = run(layer, x, bank, p, ops.mix(prev_bank, prev_x, self.msg_q[j], 4))
        for layer in L[LOOP1:]:
            x, bank = run(layer, x, bank, 0)
        return self._readout(ops.mix(bank, x, self.final_q, 1, weight=self.norm, dtype=BF)[0])

    def _fused_visits(self):
        layers = self.layers
        x, bank = self.st.x, _DecodeBank(self.decode_bank, 0, 1)
        attention_input = None

        def run_range(start, stop, p, *, previous=None):
            nonlocal x, bank, attention_input
            for layer in layers[start:stop]:
                if self.taps is not None and (tap := MTP_TAP_VISITS.get((layer.i, p))) is not None:
                    ttnn.copy(x, self.taps[tap])
                messages = None
                if previous is not None:
                    prev_x, prev_bank = previous
                    # Messages skip the shared prefix and the previous pass's entry row.
                    messages = rowvm.attnres_mix(
                        prev_bank.tensor,
                        prev_x,
                        self.dec_msg_q[layer.i - LOOP0][2],
                        prev_bank.count - 2,
                        bank_start=prev_bank.tail_start + 1,
                    )
                visit = self.visits[visit_slot(layer.i, p if layer.is_loop else 0)]
                x, bank, attention_input = layer._forward(
                    x,
                    bank,
                    visit,
                    self.st,
                    messages,
                    attention_input=attention_input,
                    next_mix=self._next_mix(layer, p),
                )

        run_range(0, LOOP0, 0)
        x_pre, bank_pre = x, bank
        run_range(LOOP0, LOOP1, 0)
        if len(layers) > LOOP0:
            loop_blocks = (LOOP1 - LOOP0) // NB
            for p in range(1, NPASS):
                previous = x, bank
                x, bank, attention_input = x_pre, bank_pre._replace(tail_start=1 + p * loop_blocks), None
                run_range(LOOP0, LOOP1, p, previous=previous)
            run_range(LOOP1, len(layers), 0)
        if attention_input is None:
            attention_input = self.ops.mix(bank, x, self.dec_final_q, 1, weight=self.dec_norm, dtype=BF)[0]
        return attention_input

    def _next_mix(self, layer, pass_id):
        next_i = layer.i + 1
        if next_i == len(self.layers):
            if not layer.is_loop or pass_id == NPASS - 1:
                return self.dec_final_q[2], self.dec_norm
        elif (next_i != LOOP1 or pass_id == NPASS - 1) and not self.layers[next_i].is_global:
            after = self.layers[next_i]
            weight = None if pass_id and after.is_loop else after.dec.norm_in
            return after.dec.q_attn[2], weight
        return None

    def _readout(self, h):
        """Logits leave the graph row-major (host untilize of the tiled 201k row costs ~9 ms; device ~0.04 ms).
        Padded vocab columns are masked so the device argmax can never pick them."""
        if self.taps is not None:
            ttnn.copy(h, self.taps[3])
        if hasattr(self, "head_logits"):
            # One program copies the head's native shards into the row and reduces the masked argmax.
            rowvm.head_tail(_linear_to_shards(h, self.head), self.head_logits, self.head_index, VOCAB)
            self.next_id = self.head_index
            if hasattr(self, "embed_table"):
                st = self.st
                rowvm.token_feedback(
                    self.head_index, self.embed_table, st.x, st.pos_g, self.tau_table, st.tau, self.id_history
                )
            return self.head_logits
        logits = ttnn.to_layout(ttnn.add(dense.linear(h, self.head), self.vocab_mask), RM)
        self.next_id = ttnn.argmax(logits, dim=-1, keepdim=True)
        return logits

    def reset(self):
        for v in self.visits.values():
            for t in [v.k, v.v] + v.conv:
                ttnn.copy_host_to_device_tensor(self.c.host(torch.zeros(tuple(t.shape)), t.dtype, TILE), t)

    def slice_bank(self, bank, start):
        """Rows start..J-1 of a bank, moved to rows 0.. (the previous pass's message bank)."""
        S, J = bank
        sel = torch.zeros(1, 1, 32, 32)
        for r in range(start, J):
            sel[0, 0, r - start, r] = 1.0
        key = (start, J)
        if key not in self._shift_cache:
            self._shift_cache[key] = self.c.up(sel, FP)
        return (self.c.mm(self._shift_cache[key], S), J - start)

    def _warmup_graph(self):
        self.logits = self.graph()

    def capture(self):
        """Warm up (compiles programs, mutates states), capture, then reset states. Anything that must persist
        across replays (prefill constants included) has to be allocated before this point."""
        self.st.write(self.embed_row(0), 0)
        self._warmup_graph()
        ttnn.synchronize_device(self.c.dev)
        self.trace_id = ttnn.begin_trace_capture(self.c.dev, cq_id=0)
        self.logits = self.graph()
        ttnn.end_trace_capture(self.c.dev, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.c.dev)
        self.reset()

    def run(self, tok, pos):
        if not 0 <= pos < self.lmax:
            raise ValueError(f"decode position {pos} is outside the allocated cache [0, {self.lmax})")
        self.st.write(self.embed_row(tok), pos)
        if self.trace_id is None:
            self.logits = self.graph()
        else:
            ttnn.execute_trace(self.c.dev, self.trace_id, cq_id=0, blocking=False)
            if pos + 1 < self.lmax and self.c.host_prefetch:
                # Build the next position's host rows while the device runs this step.
                self.st._position_rows(pos + 1)

    def step(self, tok, pos):
        """One decode token; returns fp32 softcapped logits [200058]."""
        self.run(tok, pos)
        return self.read_logits()

    def read_logits(self):
        lg = self.c.read(self.logits).reshape(-1)[:VOCAB].float()
        return 30.0 * torch.tanh(lg / 30.0)

    def _launch_greedy(self, tok, pos):
        """Replay without a host write when the device already holds this step's inputs."""
        if not 0 <= pos < self.lmax:
            raise ValueError(f"decode position {pos} is outside the allocated cache [0, {self.lmax})")
        st = self.st
        if self.trace_id is None or getattr(st, "device_state", None) != (tok, pos):
            self.run(tok, pos)
        else:
            ttnn.execute_trace(self.c.dev, self.trace_id, cq_id=0, blocking=False)

    def step_greedy(self, tok, pos):
        """One decode token with on-device argmax; returns the next token id."""
        self._launch_greedy(tok, pos)
        nxt = int(self.c.read(self.next_id).reshape(-1)[0])
        if hasattr(self, "embed_table"):
            self.st.device_state = (nxt, pos + 1)
        return nxt

    def greedy(self, tok, pos, n):
        """n greedy tokens from (tok, pos) with one host write and one readback; returns the IDs."""
        if not hasattr(self, "embed_table") or self.trace_id is None:
            out = []
            for k in range(n):
                tok = self.step_greedy(tok, pos + k)
                out.append(tok)
            return out
        if pos + n >= self.lmax:
            raise ValueError("greedy run exceeds lmax")
        self._launch_greedy(tok, pos)
        for _ in range(n - 1):
            ttnn.execute_trace(self.c.dev, self.trace_id, cq_id=0, blocking=False)
        out = self.c.read(self.id_history).reshape(-1)[pos + 1 : pos + n + 1].tolist()
        self.st.device_state = (out[-1], pos + n)
        return [int(t) for t in out]
