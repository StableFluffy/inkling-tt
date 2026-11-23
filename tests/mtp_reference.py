from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from inkling_tt.model import EPS, HD, HKV, HQ, D, W
from inkling_tt.mtp_weights import RANK

BF = torch.bfloat16


def _norm(x, weight):
    x = x.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * weight).to(BF)


def _rope(x, cos, sin):
    left, right = x.float().chunk(2, dim=-1)
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1).to(BF)


@dataclass
class DraftContext:
    length: int = 0
    kv: list = field(default_factory=lambda: [None] * 3)
    last: torch.Tensor | None = None


class DraftReference:
    def __init__(self, weights, head, fold, fold_bias, block):
        self.w = weights
        self.head = head
        self.fold = fold
        self.fold_bias = fold_bias
        self.block = block
        self.alpha = weights["fuse_weight"].float().softmax(-1).to(BF)
        self.inv_freq = 1 / (1_000_000 ** (torch.arange(0, HD, 2).float() / HD))
        self.rnn_weight = weights["markov_head.rnn_proj.weight"].float()
        self.rnn_bias = weights["markov_head.rnn_proj.bias"].float()

    def _angles(self, start, n):
        angle = torch.arange(start, start + n)[:, None, None] * self.inv_freq
        return angle.cos(), angle.sin()

    def _mix(self, features, row, weight):
        return _norm((features.float() * self.alpha[row, :, None]).sum(dim=1), weight)

    def extend(self, ctx, features):
        n = len(features)
        if n == 0 or features.shape[1:] != (4, D) or features.dtype != BF:
            raise ValueError("Expected nonempty BF16 target features [tokens,4,2304]")
        cos, sin = self._angles(ctx.length, n)
        for i in range(3):
            layer = f"layers.{i}."
            attn = layer + "self_attn."
            x = self._mix(features, i, self.w[layer + "dflare_norm.weight"])
            k = F.linear(x, self.w[attn + "k_proj_ctx.weight"]).view(n, HKV, HD)
            k = _rope(_norm(k, self.w[attn + "k_norm_ctx.weight"]), cos, sin)
            v = F.linear(x, self.w[attn + "v_proj_ctx.weight"]).view(n, HKV, HD)
            if ctx.kv[i] is not None:
                old_k, old_v = ctx.kv[i]
                k, v = torch.cat((old_k, k)), torch.cat((old_v, v))
            if i != 1 and len(k) > W:
                k, v = k[-W:].clone(), v[-W:].clone()
            ctx.kv[i] = k, v
        ctx.last = self._mix(features[-1:], 3, self.w["markov_head.dflare_norm.weight"])
        ctx.length += n

    def hidden(self, ctx, anchor_embedding):
        residual = self.w["mask_embed"].expand(self.block, -1).clone()
        residual[0] = anchor_embedding
        cos, sin = self._angles(ctx.length, self.block)
        merged = residual
        for i, (prefix_k, prefix_v) in enumerate(ctx.kv):
            layer = f"layers.{i}."
            attn = layer + "self_attn."
            x = _norm(merged, self.w[layer + "input_layernorm.weight"])
            q = F.linear(x, self.w[attn + "q_proj.weight"]).view(self.block, HQ, HD)
            k = F.linear(x, self.w[attn + "k_proj.weight"]).view(self.block, HKV, HD)
            v = F.linear(x, self.w[attn + "v_proj.weight"]).view(self.block, HKV, HD)
            q = _rope(_norm(q, self.w[attn + "q_norm.weight"]), cos, sin)
            k = _rope(_norm(k, self.w[attn + "k_norm.weight"]), cos, sin)
            k = torch.cat((prefix_k, k)).transpose(0, 1).repeat_interleave(HQ // HKV, dim=0)
            v = torch.cat((prefix_v, v)).transpose(0, 1).repeat_interleave(HQ // HKV, dim=0)
            scores = q.transpose(0, 1).float() @ k.float().transpose(-2, -1)
            attention = ((scores * HD**-0.5).softmax(-1) @ v.float()).to(BF)
            output = F.linear(attention.transpose(0, 1).reshape(self.block, -1), self.w[attn + "o_proj.weight"])
            merged = residual.float() + output.float()
            residual = merged.to(BF)
            x = _norm(merged, self.w[layer + "post_attention_layernorm.weight"])
            gate = F.linear(x, self.w[layer + "mlp.gate_proj.weight"])
            up = F.linear(x, self.w[layer + "mlp.up_proj.weight"])
            mlp = (F.silu(gate.float()) * up.float()).to(BF)
            output = F.linear(mlp, self.w[layer + "mlp.down_proj.weight"])
            merged = residual.float() + output.float()
            residual = merged.to(BF)
        return _norm(merged, self.w["norm.weight"])

    def score(self, hidden, last, anchor):
        base = F.linear(hidden, self.head)
        fixed = F.linear(
            torch.cat((hidden, last.expand_as(hidden)), -1).float(), self.rnn_weight[:, 2 * RANK :], self.rnn_bias
        )
        state = torch.zeros(RANK, dtype=BF)
        prev = anchor
        ids, logits = [], []
        for context, base_row in zip(fixed, base):
            prev_embed = self.w["markov_head.markov_w1.weight"][prev]
            raw = F.linear(torch.cat((state, prev_embed)).float(), self.rnn_weight[:, : 2 * RANK], context)
            gate, cand, out = raw.to(BF).chunk(3)
            gate = gate.float().sigmoid().to(BF)
            state = gate * state + (1 - gate) * cand.float().tanh().to(BF)
            correction = F.linear(out.float().tanh().to(BF), self.fold, self.fold_bias)
            row = (30 * torch.tanh((base_row + correction).float() / 30)).to(BF)
            prev = int(row.argmax())
            ids.append(prev)
            logits.append(row)
        return ids, torch.stack(logits)
