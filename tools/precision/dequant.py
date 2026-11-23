"""Read one Inkling-2-Mini layer without materializing the full checkpoint."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

CHECKPOINT = Path(os.environ.get("INKLING_SNAPSHOT", "Inkling-2-Mini"))
SHARD = "model-00001-of-00001.safetensors"
PROJECTIONS = {"gate_up_proj": "w13", "down_proj": "w2"}
SUFFIXES = ("weight", "weight_scale", "weight_amax", "weight_outer_scale_denominator")
E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)
_BYTE = torch.arange(256)
E2M1_PAIRS = torch.stack((E2M1[_BYTE & 15], E2M1[_BYTE >> 4]), dim=-1)
HF_CHECK_LAYERS = (0, 20)


@dataclass
class Decoded:
    fp32: torch.Tensor
    bf16: torch.Tensor
    losses: dict[str, int]


def layer_prefix(layer):
    return f"model.llm.layers.{layer}."


def read_raw(shard, layer, projection, experts=slice(None)):
    prefix = layer_prefix(layer) + f"mlp.experts.{PROJECTIONS[projection]}_"
    weight = shard.get_slice(prefix + "weight")[experts]
    scale = shard.get_slice(prefix + "weight_scale")[experts]
    if weight.dtype == torch.uint8:
        amax = shard.get_slice(prefix + "weight_amax")[experts]
        denominator = shard.get_tensor(prefix + "weight_outer_scale_denominator")
        return weight, scale, amax, denominator
    assert weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.uint8
    return weight, scale


def decode(raw, projection, *, loader_rounding=False):
    weight, scale, *outer = raw
    if outer:
        amax, denominator = outer
        # Low nibble is the even input channel. The inner product is exact in FP32.
        pairs = E2M1_PAIRS[weight.long()]
        inner = pairs.view(*scale.shape, 16) * scale.float()[..., None]
        # Compute the checkpoint rational before rounding to FP32. The numerator
        # has at most 31 significant bits. FP64 precision resolves FP32 midpoints
        # for the stored 2688 denominator (including exact ties).
        assert denominator.numel() == 1 and denominator.item() == 2688
        native = inner.double().mul_(amax.double()[:, None, None, None]).div_(denominator.double())
        exact_fp32 = native.float()
        hf_amax = amax.bfloat16().float() if loader_rounding else amax.float()
        hf_fp32 = inner * (hf_amax / denominator.float())[:, None, None, None]
    else:
        assert not (scale == 255).any()
        values = weight.float().view(*scale.shape, 32)
        exponent = scale.float() - 127
        native = values.double() * torch.exp2(exponent.double())[..., None]
        exact_fp32 = native.float()
        hf_fp32 = values * torch.exp2(exponent)[..., None]

    assert torch.isfinite(exact_fp32).all()
    hf_bf16 = hf_fp32.to(torch.bfloat16)
    losses = {
        "elements": exact_fp32.numel(),
        "nonzero": int(torch.count_nonzero(native)),
        "native_bf16_loss": int(torch.count_nonzero(hf_bf16.double() != native)),
        "native_fp32_loss": int(torch.count_nonzero(exact_fp32.double() != native)),
        "hf_fp32_vs_reference": int(torch.count_nonzero(hf_fp32 != exact_fp32)),
        "hf_bf16_vs_rounded_fp32_bf16": int(torch.count_nonzero(hf_bf16 != exact_fp32.bfloat16())),
    }
    fp32, bf16 = (t.flatten(-2) for t in (exact_fp32, hf_bf16))
    if projection == "gate_up_proj":
        fp32, bf16 = (torch.cat((t[:, ::2, :], t[:, 1::2, :]), dim=1) for t in (fp32, bf16))
    return Decoded(fp32, bf16, losses)


def hf_reference(shard, layer, projection, *, loader_rounding=False):
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    # The outer model dispatches these converters to its Inkling2Loop text model.
    (converter,) = [
        op
        for op in get_checkpoint_conversion_mapping("inkling2_loop_text")
        if op.target_patterns == [f"mlp.experts.{projection}"]
    ]
    raw = read_raw(shard, layer, projection)
    if loader_rounding:
        from transformers.core_model_loading import convert_and_load_state_dict_in_model
        from transformers.modeling_utils import LoadStateDictConfig
        from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

        # Exercise HF's complete generic loading path on just this parameter.
        model = torch.nn.Module()
        model.base_model_prefix = "model"
        model.config = Inkling2LoopTextConfig()
        model.mlp = torch.nn.Module()
        model.mlp.experts = torch.nn.Module()
        n, rows, width = raw[0].shape
        width = width * 2 if len(raw) == 4 else width
        target = torch.empty(n, rows, width, device="meta", dtype=torch.bfloat16)
        setattr(model.mlp.experts, projection, torch.nn.Parameter(target))
        prefix = "mlp.experts." + PROJECTIONS[projection] + "_"
        source = {prefix + key: tensor for key, tensor in zip(SUFFIXES, raw)}
        config = LoadStateDictConfig(dtype=torch.bfloat16, weight_mapping=[converter])
        info, _ = convert_and_load_state_dict_in_model(model, source, config)
        assert not (info.missing_keys or info.unexpected_keys or info.mismatched_keys or info.conversion_errors)
        return getattr(model.mlp.experts, projection).detach()
    values = dict(zip(converter.source_patterns, raw))
    for op in converter.operations:
        values = op.convert(values, converter.source_patterns, converter.target_patterns)
    result = values[converter.target_patterns[0]]
    assert result.dtype == torch.bfloat16
    return result


def load_layer(shard, layer, verify=False, chunk=4, *, loader_rounding=True):
    """Return the two full [expert, out, in] projections in FP32 and HF BF16."""
    result = {}
    for projection in PROJECTIONS:
        ref = hf_reference(shard, layer, projection, loader_rounding=loader_rounding) if verify else None
        pieces = []
        losses = {}
        for i in range(0, 128, chunk):
            experts = slice(i, i + chunk)
            decoded = decode(read_raw(shard, layer, projection, experts), projection, loader_rounding=loader_rounding)
            if ref is not None:
                assert torch.equal(decoded.bf16.view(torch.int16), ref[experts].view(torch.int16))
            pieces.append(decoded)
            for key, value in decoded.losses.items():
                losses[key] = losses.get(key, 0) + value
        result[projection] = Decoded(
            torch.cat([piece.fp32 for piece in pieces]),
            torch.cat([piece.bf16 for piece in pieces]),
            losses,
        )
        width = 2304 if projection == "gate_up_proj" else 1152
        assert result[projection].fp32.shape == (128, 2304, width)
    return result


def limit_torch(threads):
    if not 1 <= threads <= 24:
        raise ValueError("Use 1 to 24 threads")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layer", type=int, choices=range(24))
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--stored-companions",
        action="store_true",
        help="Give the converter the original FP32 amax instead of generic HF BF16 loading",
    )
    args = parser.parse_args()
    limit_torch(args.threads)
    with safe_open(args.checkpoint / SHARD, framework="pt", device="cpu") as shard:
        verify = args.layer in HF_CHECK_LAYERS
        for projection, weights in load_layer(
            shard, args.layer, verify=verify, loader_rounding=not args.stored_companions
        ).items():
            print(projection, list(weights.fp32.shape), weights.losses, "HF bitwise checked:", verify)
