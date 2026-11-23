"""Inkling 2 Mini text reference, following pinned Transformers eager inference.

Dependencies: torch and safetensors. Pass a local HF snapshot directory; see
docs/model.md for source revisions and differences from native kernels.
Inputs are rectangular batches of unpadded token IDs at contiguous positions.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequant_expert(weight, scale, amax=None, denominator=None):
    """Decode one expert or a stack to FP32 before the caller's BF16 cast."""
    if amax is None:
        blocks = weight.float().reshape(*scale.shape, 32)
        # uint8 is the exponent byte itself, including exponent -127 at code 0.
        factors = torch.exp2(scale.float() - 127).masked_fill(scale == 255, float("nan"))
        return (blocks * factors.unsqueeze(-1)).flatten(-2)
    if denominator is None:
        raise ValueError("NVFP4 requires an outer scale denominator")
    packed = weight.long()
    table = torch.tensor(E2M1, device=weight.device, dtype=torch.float32)
    values = torch.stack((table[packed & 15], table[packed >> 4]), dim=-1)
    values = values.reshape(*scale.shape, 16) * scale.float().unsqueeze(-1)
    outer = (amax.float() / denominator.float().reshape(())).reshape(*weight.shape[:-2], 1, 1, 1)
    return (values * outer).flatten(-2)


def rms(x, eps, weight=None):
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + eps)
    return y if weight is None else y * weight.float()


def l2_clip(x, limit, scale=None):
    y = x.float()
    if scale is not None:
        y = y * scale.float()
    norm = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    return (y * (limit / (norm + 1e-6)).clamp(max=1)).to(x.dtype)


def softcap(x, cap):
    return (x / cap).tanh() * cap


def depth_mix(states, queries):
    # AttnRes and cross-pass message reads attend over depth independently per token.
    bank = rms(torch.stack(states, dim=-2), 1e-6)
    queries = queries.reshape(-1, bank.shape[-1])
    probs = ((queries @ bank.transpose(-1, -2)) * bank.shape[-1] ** -0.5).softmax(-1)
    return (probs @ bank).unbind(-2)


@dataclass
class LayerState:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    conv: tuple = (None, None, None, None)


@dataclass
class Cache:
    """States indexed by HF cache slot, with one absolute position for the batch."""

    layers: tuple[LayerState, ...]
    length: int = 0


class InklingText:
    def __init__(self, config, weights, *, hf_loader_rounding=False):
        self.cfg = c = config
        self.w = weights
        self.device = weights["embed.weight"].device
        self.dtype = weights["embed.weight"].dtype
        self.hf_loader_rounding = hf_loader_rounding
        self.eps = c["rms_norm_eps"]
        self.width = c["hidden_size"]
        self.local = set(c["local_layer_ids"])
        self.num_layers = c["num_hidden_layers"]
        self.kernel = c["sconv_kernel_size"]
        self.window = c["sliding_window_size"]
        loop = c["loop_config"]
        self.start, self.end, self.passes = loop["core_start"], loop["core_end"], loop["num_passes"]
        if loop["is_varloop"]:
            raise ValueError("This reference is for the fixed-loop Mini")
        self.num_slots = self.num_layers + (self.passes - 1) * (self.end - self.start)
        q = weights["middle_cycle_cross_pass_queries"].reshape(-1, 2, self.width)
        kv = weights["middle_cycle_cross_pass_kv_queries"].reshape(-1, 2, self.width)
        self.message_queries = torch.cat((q[:, :1], kv, q[:, 1:]), dim=1)

    @classmethod
    def from_pretrained(cls, folder, device="cpu", *, hf_loader_rounding=False):
        """Load main text tensors. Packed experts stay CPU-mapped and decode on use."""
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text())["text_config"]
        index = folder / "model.safetensors.index.json"
        shards = (
            sorted(set(json.loads(index.read_text())["weight_map"].values()))
            if index.exists()
            else sorted(p.name for p in folder.glob("model*.safetensors"))
        )
        if not shards:
            raise FileNotFoundError(f"No main checkpoint shards in {folder}")
        weights = {}
        for shard in shards:
            with safe_open(folder / shard, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118 -- safe_open handles expose keys(), not __iter__
                    if key.startswith("model.llm."):
                        tensor = f.get_tensor(key)
                        name = key.removeprefix("model.llm.")
                        if ".mlp.experts." not in name:
                            tensor = tensor.to(device)
                        weights[name] = tensor
        return cls(config, weights, hf_loader_rounding=hf_loader_rounding)

    def _control(self, prefix, dest, message):
        w = self.w
        pair = torch.cat((rms(dest, self.eps), rms(message, self.eps)), dim=-1)
        d = w[prefix + ".down_weight"]
        h = F.linear(pair.to(d.dtype), d).float()
        # Native vLLM/SGLang use BF16 operands with FP32 output for the up GEMM.
        forget = F.linear(h, w[prefix + ".forget_weight"].float()).tanh()
        gain = F.linear(h, w[prefix + ".gain_weight"].float()).tanh()
        return (1 - forget) * dest.float() + gain * message.float()

    def _convolve(self, prefix, x, past, dynamic_input=None, heads=None):
        tail = self.kernel - 1
        x32 = x.float()
        if past is None:
            past = x32.new_zeros(x.shape[0], tail, x.shape[-1])
        ctx = torch.cat((past, x32), dim=1)
        if dynamic_input is None:
            # HF upcasts static conv weights and inputs. Checkpoint taps are oldest first.
            filt = self.w[prefix + ".weight"].float()
            y = F.conv1d(ctx.transpose(1, 2).contiguous(), filt, groups=x.shape[-1]).transpose(1, 2)
            y = (y + x32) / math.sqrt(self.kernel + 1)
        else:
            # Dynamic checkpoint taps are lag-indexed (current token first).
            taps = F.linear(dynamic_input, self.w[prefix + ".weight_proj"]).float()
            taps = taps.reshape(*x.shape[:2], heads, 1, self.kernel)
            static = self.w[prefix + ".static_w"].float().T.reshape(heads, -1, self.kernel)
            windows = ctx.unfold(1, self.kernel, 1).flip(-1).reshape(*x.shape[:2], heads, -1, self.kernel)
            y = (windows * (static + taps)).sum(-1).reshape_as(x)
            y = (y + x32) / math.sqrt(2 * self.kernel + 1)
        return y.to(x.dtype), ctx[:, -tail:].clone()

    def _attention(self, prefix, inputs, local, positions, old):
        w, c = self.w, self.cfg
        qin, kin, vin = inputs
        b, t, _ = qin.shape
        kind = "swa_" if local else ""
        nq = c[kind + "num_attention_heads"]
        nk = c[kind + "num_key_value_heads"]
        d = c[kind + "head_dim"]
        k, k_conv = self._convolve(
            prefix + "k_sconv",
            F.linear(kin, w[prefix + "wk_dv.weight"]),
            old.conv[0],
            None if local else kin,
            nk,
        )
        v, v_conv = self._convolve(
            prefix + "v_sconv",
            F.linear(vin, w[prefix + "wv_dv.weight"]),
            old.conv[1],
            None if local else vin,
            nk,
        )
        limit = c["qk_l2_clip_threshold"]
        k = l2_clip(k.reshape(b, t, nk, d), limit).transpose(1, 2)
        v = rms(v.reshape(b, t, nk, d), self.eps).to(v.dtype).transpose(1, 2)
        q = F.linear(qin, w[prefix + "wq_du.weight"]).reshape(b, t, nq, d)
        r = F.linear(qin, w[prefix + "wr_du.weight"]).reshape(b, t, nq, c["d_rel"])
        q = l2_clip(q, limit, w[prefix + "q_l2_clip.scale"])
        r = l2_clip(r, c["rel_l2_clip_threshold"], w[prefix + "rel_l2_clip.scale"])
        if not local:
            # HF rounds the clipped tensors to BF16 before applying the FP32 tau.
            ratio = (positions + 1).float() / c["log_scaling_n_floor"]
            tau = 1 + c["log_scaling_alpha"] * ratio.clamp_min(1).log()
            q = (q.float() * tau[:, None, None]).to(q.dtype)
            r = (r.float() * tau[:, None, None]).to(r.dtype)

        if old.key is not None:
            k = torch.cat((old.key, k), dim=-2)
            v = torch.cat((old.value, v), dim=-2)
        key_positions = positions[-1] + 1 - k.shape[-2] + torch.arange(k.shape[-2], device=q.device)
        distance = positions[:, None] - key_positions
        extent = self.window if local else c["rel_extent"]
        table = (r @ w[prefix + "rel_logits_proj.proj"]).transpose(1, 2)
        rel_index = distance.clamp(0, extent - 1).expand(b, nq, -1, -1)
        bias = table.gather(-1, rel_index).masked_fill((distance < 0) | (distance >= extent), 0)
        kr = k.repeat_interleave(nq // nk, dim=1)
        vr = v.repeat_interleave(nq // nk, dim=1)
        # HF eager rounds matmul/scale/add at the BF16 boundaries; fused paths differ.
        logits = (q.transpose(1, 2) @ kr.transpose(-1, -2)) * (d**-0.5) + bias
        invisible = distance < 0
        if local:
            invisible = invisible | (distance >= self.window)
        logits = logits.masked_fill(invisible, torch.finfo(logits.dtype).min)
        sink = F.linear(qin, w[prefix + "attn_sink_proj.weight"]).transpose(1, 2).unsqueeze(-1)
        probs = torch.cat((logits, sink), dim=-1).float().softmax(-1)[..., :-1].to(q.dtype)
        y = (probs @ vr).transpose(1, 2).reshape(b, t, nq * d)
        gate = F.linear(qin, w[prefix + "gate_attn.weight"]).float().sigmoid()
        y = F.linear((y.float() * gate).to(y.dtype), w[prefix + "wo_ud.weight"])
        if local:
            # Retain only prior keys needed by the next query, with bounded backing storage.
            k = k[:, :, -(self.window - 1) :].clone() if self.window > 1 else k[:, :, :0].clone()
            v = v[:, :, -(self.window - 1) :].clone() if self.window > 1 else v[:, :, :0].clone()
        return y, k, v, k_conv, v_conv

    def _expert_weight(self, prefix, expert, x):
        def read(suffix):
            t = self.w[prefix + suffix]
            return (t if suffix == "_outer_scale_denominator" else t[expert]).to(x.device)

        weight, scale = read(""), read("_scale")
        if weight.dtype == torch.uint8:
            amax, denominator = read("_amax"), read("_outer_scale_denominator")
            # HF's generic loader can round F32 companions before its converter.
            # Preserve stored precision by default, as in the GGUF/native loaders.
            if self.hf_loader_rounding:
                amax, denominator = amax.to(x.dtype), denominator.to(x.dtype)
            return dequant_expert(weight, scale, amax, denominator).to(x.dtype)
        return dequant_expert(weight, scale).to(x.dtype)

    def _moe(self, prefix, x):
        w, c = self.w, self.cfg
        shape = x.shape
        x = x.reshape(-1, self.width)
        scores = F.linear(x.float(), w[prefix + "gate.weight"].float()).sigmoid()
        ids = (scores + w[prefix + "gate.bias"]).topk(c["num_experts_per_tok"], sorted=False).indices
        gamma = scores.gather(-1, ids)
        gamma = gamma / (gamma.sum(-1, keepdim=True) + 1e-20) * c["route_scale"]
        out = torch.zeros_like(x, dtype=torch.float32)
        # HF dequantizes to BF16 and applies gamma after w2. Native vLLM routes
        # gamma into SwiGLU before re-quantizing for w2; see the spec.
        for expert in ids.unique(sorted=True).tolist():
            token, slot = torch.where(ids == expert)
            w13 = self._expert_weight(prefix + "experts.w13_weight", expert, x)
            # Stored w13 rows alternate gate, up. Unweave before the HF-style GEMM.
            w13 = w13.reshape(-1, 2, self.width).transpose(0, 1).reshape_as(w13).contiguous()
            g, u = F.linear(x[token], w13).float().chunk(2, dim=-1)
            y = softcap(g, c["mlp_gate_softcap"]) * g.sigmoid() * softcap(u, c["mlp_up_softcap"])
            w2 = self._expert_weight(prefix + "experts.w2_weight", expert, x)
            y = F.linear(y.to(x.dtype), w2).float() * gamma[token, slot, None]
            out.index_add_(0, token, y)
        return out.to(x.dtype).reshape(shape)

    def _layer(self, i, partial, blocks, messages, positions, old):
        w = self.w
        p = f"layers.{i}."
        local = i in self.local
        inputs = (partial,) if i == 0 else depth_mix([*blocks, partial], w[f"attn_res.attn_queries.{i - 1}"])
        if local:
            inputs = inputs * 3
        norms = ("attn_norm",) * 3 if local else ("q_input_norm", "k_input_norm", "v_input_norm")
        if messages is not None:
            inputs = tuple(
                self._control(p + "controllers." + stream, x, m)
                for stream, x, m in zip(("attn", "k", "v"), inputs, messages[:3])
            )
        inputs = tuple(rms(x, self.eps, w[p + n + ".weight"]).to(self.dtype) for x, n in zip(inputs, norms))
        y, k, v, hc0, hc1 = self._attention(p + "attn.", inputs, local, positions, old)
        y, hc2 = self._convolve(p + "attn_sconv", y, old.conv[2])
        if i % self.cfg["n_layers_per_attn_res_block"] == 0:
            blocks = [*blocks, partial]
            partial = y.float()
        else:
            partial = partial.float() + y.float()
        (mlp,) = depth_mix([*blocks, partial], w[f"attn_res.mlp_queries.{i}"])
        if messages is not None:
            mlp = self._control(p + "controllers.mlp", mlp, messages[3])
        mlp = rms(mlp, self.eps, w[p + "mlp_norm.weight"]).to(self.dtype)
        y = self._moe(p + "mlp.", mlp)
        y, hc3 = self._convolve(p + "mlp_sconv", y, old.conv[3])
        return (partial + y.float()).to(self.dtype), blocks, LayerState(k, v, (hc0, hc1, hc2, hc3))

    @torch.inference_mode()
    def forward(self, input_ids, cache=None, *, last_only=False, trace=None):
        """Prefill or append tokens. Return (FP32 logits, new cache).

        A trace callback receives (visit, physical_layer, pass, partial) after
        each visit. Input cache remains available for branching or comparison.
        """
        if input_ids.ndim != 2 or input_ids.numel() == 0 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a nonempty [batch, time] LongTensor")
        if cache is None:
            cache = Cache(tuple(LayerState() for _ in range(self.num_slots)))
        elif len(cache.layers) != self.num_slots:
            raise ValueError("Cache slot layout does not match the model")
        if cache.length and cache.layers[0].key.shape[0] != input_ids.shape[0]:
            raise ValueError("Cache batch size does not match input_ids")
        w = self.w
        positions = torch.arange(cache.length, cache.length + input_ids.shape[1], device=self.device)
        x = F.embedding(input_ids.to(self.device), w["embed.weight"])
        x = rms(x, self.eps, w["embed_norm.weight"]).to(self.dtype)
        slots = list(cache.layers)
        visit = 0

        def run(begin, end, pass_idx, partial, blocks, bank=None):
            nonlocal visit
            offset = 0
            if pass_idx > 0:
                offset = self.num_layers + (pass_idx - 1) * (self.end - self.start) - self.start
            for i in range(begin, end):
                slot = i + offset
                messages = None if bank is None else depth_mix(bank, self.message_queries[i - self.start])
                partial, blocks, slots[slot] = self._layer(i, partial, blocks, messages, positions, slots[slot])
                if trace is not None:
                    trace(visit, i, pass_idx, partial)
                visit += 1
            return partial, blocks

        pre_x, pre_blocks = run(0, self.start, 0, x, [])
        x, blocks = run(self.start, self.end, 0, pre_x, pre_blocks)
        for p in range(1, self.passes):
            bank = [*blocks[len(pre_blocks) + 1 :], x]
            x, blocks = run(self.start, self.end, p, pre_x, pre_blocks, bank)
        x, blocks = run(self.end, self.num_layers, 0, x, blocks)

        if last_only:
            blocks, x = [block[:, -1:] for block in blocks], x[:, -1:]
        (x,) = depth_mix([*blocks, x], w["attn_res.output_query"])
        x = rms(x, self.eps, w["norm.weight"]).to(self.dtype)
        # HF GEMM returns BF16 before .float(); vLLM/SGLang use an FP32 GEMM epilogue.
        logits = F.linear(x, w["unembed.weight"][: self.cfg["unpadded_vocab_size"]]).float()
        updated = Cache(tuple(slots), cache.length + input_ids.shape[1])
        return softcap(logits, self.cfg["final_logit_softcapping"]), updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("ids", type=int, nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1, help="number of greedy tokens to emit")
    parser.add_argument(
        "--hf-loader-rounding",
        action="store_true",
        help="round NVFP4 outer scale companions as in pinned Transformers BF16 loading",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_float32_matmul_precision("highest")
    model = InklingText.from_pretrained(args.snapshot, args.device, hf_loader_rounding=args.hf_loader_rounding)
    ids = torch.tensor([args.ids], dtype=torch.long)
    cache = None
    for _ in range(args.steps):
        logits, cache = model.forward(ids, cache, last_only=True)
        ids = logits[:, -1:].argmax(-1)
        print(ids.item(), flush=True)


if __name__ == "__main__":
    main()
