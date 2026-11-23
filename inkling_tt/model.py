"""Inkling-2-Mini text decoder on Tenstorrent (TTNN), milestone 1: batch 1, one token per step.

Every op of the HF eager reference (transformers 5.24, models/inkling2_loop) is mirrored, including its fp32/bf16
cast points. Positional bookkeeping (ring slots, relative-distance indices, masks, log-scale factor) is computed on
the host and uploaded per step; routing is computed on the host in fp32 from the device activation.
"""

import math
from dataclasses import dataclass

import torch
import ttnn
from safetensors import safe_open

D, HQ, HKV, HD, DR, F, E, TOPK = 2304, 24, 8, 128, 16, 1152, 128, 8
GQ = HQ // HKV
W = 1024
NB = 6
LOOP0, LOOP1, NPASS = 6, 18, 3
GLOBAL_LAYERS = (5, 11, 17, 23)
QK_CLIP = 22.627416997969522
REL_CLIP = 8.0
LOG_FLOOR, LOG_ALPHA = 8192, 0.1
EPS = 1e-6
NEG = -1e30
VOCAB = 200058


def visit_slot(layer, pass_idx):
    if pass_idx == 0:
        return layer
    return 24 + (pass_idx - 1) * (LOOP1 - LOOP0) + (layer - LOOP0)


@dataclass
class Precision:
    expert_dtype: object = None
    dense_dtype: object = None
    expert_fidelity: object = None
    dense_fidelity: object = None


class Ctx:
    def __init__(self, device, prec: Precision):
        self.dev = device
        self.prec = prec
        arch = device.arch()
        self.ck_exact = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.ck_dense = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=prec.dense_fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.ck_expert = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=prec.expert_fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def up(self, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def up_rows(self, v, dtype=ttnn.bfloat16):
        """A vector broadcast to 32 rows, [1,1,32,n]."""
        return self.up(v.reshape(1, 1, 1, -1).expand(1, 1, 32, -1).contiguous().float(), dtype)

    def lin(self, x, w, ck=None, dtype=ttnn.bfloat16):
        return ttnn.linear(x, w, compute_kernel_config=ck or self.ck_dense, dtype=dtype)

    def mm(self, a, b, dtype=ttnn.float32, transpose_b=False):
        return ttnn.matmul(a, b, transpose_b=transpose_b, compute_kernel_config=self.ck_exact, dtype=dtype)


def f32(x):
    return x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)


def bf16(x):
    return x if x.dtype == ttnn.bfloat16 else ttnn.typecast(x, ttnn.bfloat16)


class Ops:
    """Shared constant tensors and composite ops."""

    def __init__(self, c: Ctx):
        self.c = c
        self.seg = {}

    def segment(self, nheads, hd):
        """Block-diagonal ones [nheads*hd, 32] and its transpose for per-head reductions/broadcasts."""
        key = (nheads, hd)
        if key not in self.seg:
            m = torch.zeros(nheads * hd, 32)
            for h in range(nheads):
                m[h * hd : (h + 1) * hd, h] = 1.0
            self.seg[key] = (self.c.up(m, ttnn.float32), self.c.up(m.T.contiguous(), ttnn.float32))
        return self.seg[key]

    def rms(self, x, w=None):
        """fp32 RMSNorm over the last dim; w is a [1,1,32,n] fp32 row tensor or None."""
        x = f32(x)
        ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
        y = ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ms, EPS)))
        return y if w is None else ttnn.multiply(y, w)

    def head_sumsq(self, x, nheads, hd):
        s, _ = self.segment(nheads, hd)
        return self.c.mm(ttnn.multiply(x, x), s)

    def head_bcast(self, v, nheads, hd):
        _, st = self.segment(nheads, hd)
        return self.c.mm(v, st)

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

    def softcap(self, x, cap):
        return ttnn.multiply(ttnn.tanh(ttnn.multiply(x, 1.0 / cap)), cap)

    def mix(self, nbank, partial, qmat, nq):
        """AttnRes depth mixing for one token per row. nbank: normalized fp32 bank vectors; qmat [2304,32] fp32
        holding nq query columns pre-scaled by 1/sqrt(2304). Returns nq fp32 tensors [1,1,32,2304]."""
        items = nbank + [self.rms(partial)]
        logits = [self.c.mm(n, qmat) for n in items]
        m = logits[0]
        for l in logits[1:]:
            m = ttnn.maximum(m, l)
        ex = [ttnn.exp(ttnn.subtract(l, m)) for l in logits]
        s = ex[0]
        for e in ex[1:]:
            s = ttnn.add(s, e)
        rs = ttnn.reciprocal(s)
        outs = []
        for q in range(nq):
            acc = None
            for n, e in zip(items, ex):
                p = ttnn.slice(ttnn.multiply(e, rs), [0, 0, 0, q], [1, 1, 32, q + 1])
                t = ttnn.multiply(n, p)
                acc = t if acc is None else ttnn.add(acc, t)
            outs.append(acc)
        return outs


def rows0_to_groups(x, width_per_head, padw=0):
    """[1,1,32,24*w] (row 0 valid) -> [1,8,32,w+padw] with rows 0..2 = the three query heads of each KV group."""
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.slice(x, [0, 0, 0, 0], [1, 1, 1, HQ * width_per_head])
    x = ttnn.reshape(x, [1, HKV, GQ, width_per_head])
    x = ttnn.pad(x, [(0, 0), (0, 0), (0, 32 - GQ), (0, padw)], 0.0)
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


def groups_to_rows0(y, width_per_head):
    y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
    y = ttnn.slice(y, [0, 0, 0, 0], [1, HKV, GQ, width_per_head])
    y = ttnn.reshape(y, [1, 1, 1, HQ * width_per_head])
    y = ttnn.pad(y, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
    return ttnn.to_layout(y, ttnn.TILE_LAYOUT)


class Weights:
    def __init__(self, path):
        self.path = path
        import json

        with open(f"{path}/index.json") as f:
            self.index = json.load(f)
        self.handles = {}

    def get(self, name):
        g = self.index[name][0]
        if g not in self.handles:
            self.handles[g] = safe_open(f"{self.path}/{g}.safetensors", framework="pt")
        return self.handles[g].get_tensor(name)


class ConvState:
    """Three previous rows (t-1, t-2, t-3) of one causal depthwise conv, fp32 [1,1,32,C]."""

    def __init__(self, c: Ctx, ch):
        z = torch.zeros(1, 1, 32, ch)
        self.h = [c.up(z, ttnn.float32) for _ in range(3)]

    def push(self, x):
        self.h = [x] + self.h[:2]


class Layer:
    def __init__(self, c: Ctx, ops: Ops, wts: Weights, i, lmax):
        self.c, self.ops, self.i = c, ops, i
        p = f"layers.{i}."
        g = lambda n: wts.get(p + n)
        self.is_global = i in GLOBAL_LAYERS
        self.is_loop = LOOP0 <= i < LOOP1
        self.block_boundary = i % NB == 0
        dd = c.prec.dense_dtype
        T = lambda w: c.up(w.T.contiguous(), dd)
        a = "self_attn."
        self.wq = T(g(a + "q_proj.linear.weight"))
        self.qscale = c.up_rows(g(a + "q_proj.norm.scale").float().reshape(-1), ttnn.float32)
        self.wr = T(g(a + "relative_position_bias.linear.weight"))
        self.rscale = c.up_rows(g(a + "relative_position_bias.norm.scale").float().reshape(-1), ttnn.float32)
        proj = g(a + "relative_position_bias.proj").float()
        table = torch.zeros(W + 1, 32)
        table[:W, :DR] = proj.T
        self.rel_table = c.up(table, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.wsink = T(g(a + "sink_proj.weight"))
        self.wgate = T(g(a + "o_proj.gate_proj.weight"))
        self.wo = T(g(a + "o_proj.linear.weight"))
        self.wk = T(g(a + "k_proj.linear.weight"))
        self.wv = T(g(a + "v_proj.linear.weight"))
        if self.is_global:
            self.k_static = [c.up_rows(r, ttnn.float32) for r in g(a + "k_proj.sconv.static_weight").float()]
            self.v_static = [c.up_rows(r, ttnn.float32) for r in g(a + "v_proj.sconv.static_weight").float()]
            self.k_wproj = T(g(a + "k_proj.sconv.weight_proj.weight"))
            self.v_wproj = T(g(a + "v_proj.sconv.weight_proj.weight"))
            self.dyn_sel = []
            for lag in range(4):
                m = torch.zeros(32, HKV * HD)
                for h in range(HKV):
                    m[h * 4 + lag, h * HD : (h + 1) * HD] = 1.0
                self.dyn_sel.append(c.up(m, ttnn.float32))
            self.norm_q = c.up_rows(g("q_input_layernorm.weight").float(), ttnn.float32)
            self.norm_k = c.up_rows(g("k_input_layernorm.weight").float(), ttnn.float32)
            self.norm_v = c.up_rows(g("v_input_layernorm.weight").float(), ttnn.float32)
        else:
            self.k_taps = self._taps(g(a + "k_proj.sconv.conv1d.weight"))
            self.v_taps = self._taps(g(a + "v_proj.sconv.conv1d.weight"))
            self.norm_in = c.up_rows(g("input_layernorm.weight").float(), ttnn.float32)
        self.norm_post = c.up_rows(g("post_attention_layernorm.weight").float(), ttnn.float32)
        self.pa_taps = self._taps(g("post_attention_convolution.conv1d.weight"))
        self.pm_taps = self._taps(g("post_mlp_convolution.conv1d.weight"))
        self.q_attn = None
        if i > 0:
            qa = g("attention_input.query").float().reshape(-1, D)
            self.q_attn, self.nq_attn = self._qmat(qa), qa.shape[0]
        self.q_mlp = self._qmat(g("mlp_input.query").float().reshape(-1, D))
        if self.is_loop:
            self.gates = {}
            for s in ("q", "k", "v", "mlp"):
                dn = g(f"{s}_message_gate.down_proj.weight")
                self.gates[s] = (
                    T(dn),
                    c.up(g(f"{s}_message_gate.forget_gate_proj.weight").float().T.contiguous(), ttnn.float32),
                    c.up(g(f"{s}_message_gate.message_gate_proj.weight").float().T.contiguous(), ttnn.float32),
                )
        self.router_w = g("mlp.gate.weight").float()
        self.router_b = g("mlp.gate.e_score_correction_bias").float()
        gu = g("mlp.experts.gate_up_proj")
        dn = g("mlp.experts.down_proj")
        ed = c.prec.expert_dtype(i) if callable(c.prec.expert_dtype) else c.prec.expert_dtype
        self.w_gu = [c.up(gu[e].T.contiguous(), ed) for e in range(E)]
        self.w_dn = [c.up(dn[e].T.contiguous(), ed) for e in range(E)]

    def _taps(self, w):
        w = w.float().reshape(-1, 4)
        return [self.c.up_rows(w[:, j].contiguous(), ttnn.float32) for j in range(4)]

    def _qmat(self, q):
        m = torch.zeros(D, 32)
        m[:, : q.shape[0]] = q.T / math.sqrt(D)
        return self.c.up(m, ttnn.float32)

    def sconv(self, x, taps, st: ConvState):
        """Static causal conv: (x + sum_j w_j x_{t+j-3}) / sqrt(5); tap 3 is the current row."""
        x = f32(x)
        acc = ttnn.add(x, ttnn.multiply(x, taps[3]))
        for j, h in zip((2, 1, 0), st.h):
            acc = ttnn.add(acc, ttnn.multiply(h, taps[j]))
        st.push(x)
        return ttnn.multiply(acc, 1.0 / math.sqrt(5.0))

    def dsconv(self, x, xin, static, wproj, st: ConvState):
        """Dynamic causal conv on global K/V: (x + sum_lag (static_lag + dyn_lag) x_{t-lag}) / 3."""
        x = f32(x)
        dyn = f32(self.c.lin(xin, wproj))
        hist = [x] + st.h
        acc = x
        for lag in range(4):
            wl = ttnn.add(static[lag], self.c.mm(dyn, self.dyn_sel[lag]))
            acc = ttnn.add(acc, ttnn.multiply(wl, hist[lag]))
        st.push(x)
        return ttnn.multiply(acc, 1.0 / 3.0)

    def gate_msg(self, s, u, m):
        dn, wf, wg = self.gates[s]
        ops, c = self.ops, self.c
        u, m = f32(u), f32(m)
        h = c.lin(bf16(ttnn.concat([ops.rms(u), ops.rms(m)], dim=-1)), dn)
        h = f32(h)
        fg = ttnn.tanh(c.mm(h, wf))
        mg = ttnn.tanh(c.mm(h, wg))
        return ttnn.add(ttnn.multiply(ttnn.rsub(fg, 1.0), u), ttnn.multiply(mg, m))

    def attention(self, xq, xk, xv, st, pos):
        c, ops = self.c, self.ops
        tau = 1.0 + LOG_ALPHA * math.log(max((pos + 1) / LOG_FLOOR, 1.0)) if self.is_global else None
        q = bf16(ops.l2clip(c.lin(xq, self.wq), HQ, HD, QK_CLIP, self.qscale))
        r = bf16(ops.l2clip(c.lin(xq, self.wr), HQ, DR, REL_CLIP, self.rscale))
        if tau is not None:
            q = bf16(ttnn.multiply(f32(q), tau))
            r = bf16(ttnn.multiply(f32(r), tau))
        sink = c.lin(xq, self.wsink)
        gate = ttnn.sigmoid(f32(c.lin(xq, self.wgate)))
        kr, vr = c.lin(xk, self.wk), c.lin(xv, self.wv)
        if self.is_global:
            k = bf16(self.dsconv(kr, xk, self.k_static, self.k_wproj, st.conv[0]))
            v = bf16(self.dsconv(vr, xv, self.v_static, self.v_wproj, st.conv[1]))
        else:
            k = bf16(self.sconv(kr, self.k_taps, st.conv[0]))
            v = bf16(self.sconv(vr, self.v_taps, st.conv[1]))
        k = bf16(ops.l2clip(k, HKV, HD, QK_CLIP))
        v = bf16(ops.head_rms(v, HKV, HD))
        st.write_kv(k, v, pos)
        pm = st.posmeta
        qg = rows0_to_groups(q, HD)
        rg = rows0_to_groups(r, DR, 32 - DR)
        scores = c.mm(qg, st.k, transpose_b=True)
        scores = ttnn.multiply(scores, 1.0 / math.sqrt(HD))
        rel = ttnn.embedding(pm.idx, self.rel_table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        bias = c.mm(rg, rel, transpose_b=True)
        scores = ttnn.add(ttnn.add(scores, bias), pm.mask)
        sk = ttnn.to_layout(sink, ttnn.ROW_MAJOR_LAYOUT)
        sk = ttnn.reshape(ttnn.slice(sk, [0, 0, 0, 0], [1, 1, 1, HQ]), [1, HKV, GQ, 1])
        sk = ttnn.pad(f32(sk), [(0, 0), (0, 0), (0, 32 - GQ), (0, 31)], NEG)
        sk = ttnn.to_layout(sk, ttnn.TILE_LAYOUT)
        full = ttnn.concat([scores, sk], dim=-1)
        mx = ttnn.max(full, dim=-1, keepdim=True)
        ex = ttnn.exp(ttnn.subtract(full, mx))
        den = ttnn.sum(ex, dim=-1, keepdim=True)
        prob = ttnn.multiply(ex, ttnn.reciprocal(den))
        prob = bf16(ttnn.slice(prob, [0, 0, 0, 0], [1, HKV, 32, st.klen]))
        out = ttnn.matmul(prob, st.v, compute_kernel_config=c.ck_exact, dtype=ttnn.bfloat16)
        out = groups_to_rows0(out, HD)
        out = bf16(ttnn.multiply(f32(out), gate))
        return c.lin(out, self.wo)

    def moe(self, xm):
        """xm: bf16 [1,1,32,2304] (post_attention_layernorm output). Host fp32 routing, device experts."""
        c, ops = self.c, self.ops
        h = ttnn.to_torch(xm)[0, 0, 0].float()
        s = torch.sigmoid(self.router_w @ h)
        idx = torch.topk(s + self.router_b, TOPK).indices
        wsel = s[idx]
        wsel = wsel / (wsel.sum() + 1e-20) * 8.0
        acc = None
        for e, we in zip(idx.tolist(), wsel.tolist()):
            gu = c.lin(xm, self.w_gu[e], c.ck_expert)
            gt = f32(ttnn.slice(gu, [0, 0, 0, 0], [1, 1, 32, F]))
            up = f32(ttnn.slice(gu, [0, 0, 0, F], [1, 1, 32, 2 * F]))
            a = ttnn.multiply(ttnn.multiply(ops.softcap(gt, 4.0), ttnn.sigmoid(gt)), ops.softcap(up, 25.0))
            y = f32(c.lin(bf16(a), self.w_dn[e], c.ck_expert))
            y = ttnn.multiply(y, we)
            acc = y if acc is None else ttnn.add(acc, y)
        return bf16(acc), idx

    def forward(self, x, bank, st, pos, messages=None):
        """x: bf16 partial; bank: normalized fp32 completed blocks (list). Returns (x_out bf16, bank_after)."""
        ops = self.ops
        block_residual = x
        if self.is_global:
            if self.q_attn is None:
                xq = xk = xv = f32(x)
            else:
                xq, xk, xv = ops.mix(bank, x, self.q_attn, 3)
            if messages is not None:
                xq = self.gate_msg("q", xq, messages[0])
                xk = self.gate_msg("k", xk, messages[1])
                xv = self.gate_msg("v", xv, messages[2])
            xq = bf16(ops.rms(xq, self.norm_q))
            xk = bf16(ops.rms(xk, self.norm_k))
            xv = bf16(ops.rms(xv, self.norm_v))
        else:
            h = f32(x) if self.q_attn is None else ops.mix(bank, x, self.q_attn, 1)[0]
            if messages is None:
                xq = xk = xv = bf16(ops.rms(h, self.norm_in))
            else:
                xq = bf16(ops.rms(self.gate_msg("q", h, messages[0]), self.norm_in))
                xk = bf16(ops.rms(self.gate_msg("k", h, messages[1]), self.norm_in))
                xv = bf16(ops.rms(self.gate_msg("v", h, messages[2]), self.norm_in))
        a = self.attention(xq, xk, xv, st, pos)
        a = bf16(self.sconv(a, self.pa_taps, st.conv[2]))
        a = f32(a)
        if self.block_boundary:
            bank = bank + [ops.rms(block_residual)]
            z = a
        else:
            z = ttnn.add(f32(block_residual), a)
        hm = ops.mix(bank, z, self.q_mlp, 1)[0]
        if messages is not None:
            hm = self.gate_msg("mlp", hm, messages[3])
        xm = bf16(ops.rms(hm, self.norm_post))
        y, route = self.moe(xm)
        y = bf16(self.sconv(y, self.pm_taps, st.conv[3]))
        return bf16(ttnn.add(z, f32(y))), bank, route


class PosMeta:
    """Per-step relative-distance indices and additive masks shared by all layers of one attention type."""

    def __init__(self, c: Ctx, klen):
        self.c, self.klen = c, klen
        self.idx = None
        self.mask = None

    def set(self, pos, is_global):
        s = torch.arange(self.klen)
        if is_global:
            d = pos - s
            valid = d >= 0
            idx = torch.where((d >= 0) & (d < W), d, torch.full_like(d, W))
        else:
            d = (pos - s) % W
            valid = d <= pos
            idx = d
        mask = torch.where(valid, 0.0, NEG).reshape(1, 1, 1, -1).expand(1, 1, 32, -1).contiguous()
        self.idx = self.c.up(idx.reshape(1, -1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.mask = self.c.up(mask, ttnn.float32)


class VisitState:
    def __init__(self, c: Ctx, is_global, lmax, posmeta):
        self.c = c
        self.klen = lmax if is_global else W
        self.is_global = is_global
        z = torch.zeros(1, HKV, self.klen, HD)
        self.k = c.up(z)
        self.v = c.up(z)
        self.conv = [ConvState(c, HKV * HD), ConvState(c, HKV * HD), ConvState(c, D), ConvState(c, D)]
        self.posmeta = posmeta

    def write_kv(self, k, v, pos):
        slot = pos if self.is_global else pos % W
        for cache, x in ((self.k, k), (self.v, v)):
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.reshape(ttnn.slice(x, [0, 0, 0, 0], [1, 1, 1, HKV * HD]), [1, HKV, 1, HD])
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            ttnn.update_cache(cache, x, slot)


class InklingTT:
    def __init__(self, device, wpath, prec: Precision, lmax=2048, layers=None):
        self.c = Ctx(device, prec)
        self.ops = Ops(self.c)
        self.wts = Weights(wpath)
        g = self.wts.get
        self.embed = g("embed_tokens.weight")
        self.embed_norm = g("embed_tokens.embed_norm.weight").float()
        self.lmax = lmax
        n = 24 if layers is None else layers
        self.layers = [Layer(self.c, self.ops, self.wts, i, lmax) for i in range(n)]
        self.final_q = self.layers[0]._qmat(g("final_input.query").float().reshape(-1, D))
        mq = g("message_input.query").float()
        self.msg_q = [self.layers[0]._qmat(mq[j]) for j in range(LOOP1 - LOOP0)]
        self.norm = self.c.up_rows(g("norm.weight").float(), ttnn.float32)
        head = g("lm_head.weight")
        pad = (-head.shape[0]) % 32
        head = torch.cat([head, torch.zeros(pad, D, dtype=head.dtype)], 0)
        self.head = self.c.up(head.T.contiguous(), prec.dense_dtype)
        self.pm_global = PosMeta(self.c, lmax)
        self.pm_local = PosMeta(self.c, W)
        self.states = {}
        for i in range(n):
            for p in range(NPASS if LOOP0 <= i < LOOP1 else 1):
                is_g = i in GLOBAL_LAYERS
                self.states[visit_slot(i, p)] = VisitState(
                    self.c, is_g, lmax, self.pm_global if is_g else self.pm_local
                )
        self.trace = None

    def embed_row(self, tok):
        e = self.embed[tok].float()
        e = e * torch.rsqrt(e.pow(2).mean() + EPS) * self.embed_norm
        x = torch.zeros(1, 1, 32, D)
        x[0, 0, 0] = e.to(torch.bfloat16).float()
        return self.c.up(x)

    def step(self, tok, pos, record=None):
        ops = self.ops
        self.pm_global.set(pos, True)
        self.pm_local.set(pos, False)
        x = self.embed_row(tok)
        bank = []
        L = self.layers

        def run(layer, x, bank, p, messages=None):
            new_x, _, route = layer.forward(x, bank, self.states[visit_slot(layer.i, p)], pos, messages)
            if record is not None:
                record.append((layer.i, p, new_x, route))
            if layer.block_boundary:
                bank = bank + [ops.rms(x)]
            return new_x, bank

        for layer in L[:LOOP0]:
            x, bank = run(layer, x, bank, 0)
        if len(L) <= LOOP0:
            return x
        x_pre, bank_pre = x, bank
        npre = len(bank_pre)
        for layer in L[LOOP0:LOOP1]:
            x, bank = run(layer, x, bank, 0)
        for p in range(1, NPASS):
            prev_bank, prev_x = bank[npre + 1 :], x
            x, bank = x_pre, bank_pre
            for j, layer in enumerate(L[LOOP0:LOOP1]):
                msgs = ops.mix(prev_bank, prev_x, self.msg_q[j], 4)
                x, bank = run(layer, x, bank, p, msgs)
        for layer in L[LOOP1:]:
            x, bank = run(layer, x, bank, 0)
        h = ops.mix(bank, x, self.final_q, 1)[0]
        h = bf16(ops.rms(h, self.norm))
        logits = ttnn.linear(h, self.head, compute_kernel_config=self.c.ck_dense, dtype=ttnn.bfloat16)
        lg = ttnn.to_torch(logits)[0, 0, 0, :VOCAB].float()
        return 30.0 * torch.tanh(lg / 30.0)
