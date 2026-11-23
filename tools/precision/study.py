"""Measure storage error and CPU MoE sensitivity on Inkling-2-Mini."""

import argparse
import json
import math
import resource
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from bfp import NativeRowQuantizer, quantize
from dequant import (
    CHECKPOINT,
    HF_CHECK_LAYERS,
    PROJECTIONS,
    SHARD,
    decode,
    hf_reference,
    layer_prefix,
    limit_torch,
    read_raw,
)
from safetensors import safe_open

LAYERS = (0, 5, 10, 19, 20, 23)
FORMATS = {"bfp8_b": 7, "bfp4_b": 3}
STORAGE_OPTIONS = {f"{fmt}_{axis}": (bits, axis) for fmt, bits in FORMATS.items() for axis in ("N", "K")}
SEED = 1729
TOKENS = 64


def squared_norm(tensor):
    return torch.linalg.vector_norm(tensor.double()).square().item()


def update_error(stats, ref, value):
    diff = value - ref
    stats["squared_error"] += squared_norm(diff)
    stats["max_abs"] = max(stats["max_abs"], diff.abs().max().item())
    stats["changed"] += int(torch.count_nonzero(diff))


def truncate_fraction(value, kept):
    # Clear the least significant FP32 fraction bits, keeping sign and exponent.
    return (value.view(torch.int32) & -(1 << (23 - kept))).view(torch.float32)


def fidelity_linear(x, weight, mode, weight_register):
    """MVMUL phase masks with each dot product accumulated on the CPU in FP32."""
    if mode == "HiFi4":
        return F.linear(x, weight)
    a_bits, b_bits = (4 if mode == "LoFi" else 7), 6
    x_bits, w_bits = (b_bits, a_bits) if weight_register == "SrcA" else (a_bits, b_bits)
    x = x if x_bits == 7 else truncate_fraction(x, x_bits)
    weight = weight if w_bits == 7 else truncate_fraction(weight, w_bits)
    return F.linear(x, weight)


class MoeProbe:
    def __init__(self, shard, layer, config, eager_check, output_modes, storage_options):
        from transformers.models.inkling2_loop.modeling_inkling2_loop import (
            Inkling2LoopRMSNorm,
            Inkling2LoopSoftcappedSwiGLU,
            Inkling2LoopSparseMoeBlock,
            Inkling2LoopTopKRouter,
        )

        prefix = layer_prefix(layer)
        norm = Inkling2LoopRMSNorm(config.hidden_size, config.rms_norm_eps)
        norm.weight.copy_(shard.get_tensor(prefix + "mlp_norm.weight"))
        router = Inkling2LoopTopKRouter(config)
        router.weight.copy_(shard.get_tensor(prefix + "mlp.gate.weight"))
        router.e_score_correction_bias.copy_(shard.get_tensor(prefix + "mlp.gate.bias"))
        rng = torch.Generator(device="cpu").manual_seed(SEED)
        raw = torch.randn(TOKENS, config.hidden_size, generator=rng).bfloat16()
        self.inputs = norm(raw)
        _, self.route_weights, self.experts = router(self.inputs)
        assert self.route_weights.dtype == torch.float32
        self.activation = Inkling2LoopSoftcappedSwiGLU(config)
        self.routes = {
            expert: torch.where(expert == self.experts.T) for expert in sorted(set(self.experts.flatten().tolist()))
        }
        names = ["exact", "bf16", *storage_options]
        self.fidelity_cases = {
            f"{name}_{path}_{mode}": (name, mode, register)
            for name, (_, axis) in storage_options.items()
            for path, register in ((("in1", "SrcA"),) if axis == "N" else (("transB", "SrcA"), ("in0", "SrcB")))
            for mode in output_modes
        }
        self.outputs = {
            name: torch.zeros_like(self.inputs, dtype=torch.float32) for name in (*names, *self.fidelity_cases)
        }
        self.eager = None
        if eager_check:
            with torch.device("meta"):
                self.eager = Inkling2LoopSparseMoeBlock(config)
            self.eager.gate = router
            for projection in PROJECTIONS:
                shape = getattr(self.eager.experts, projection).shape
                weight = torch.empty(shape, dtype=torch.bfloat16)
                setattr(self.eager.experts, projection, torch.nn.Parameter(weight, requires_grad=False))

    def add(self, start, weights):
        ref = weights["exact"]
        if self.eager is not None:
            for projection, weight in zip(PROJECTIONS, weights["bf16"]):
                getattr(self.eager.experts, projection)[start : start + weight.shape[0]].copy_(weight)

        for offset in range(ref[0].shape[0]):
            expert = start + offset
            if expert not in self.routes:
                continue
            rank, token = self.routes[expert]
            x = self.inputs[token].float()
            routed_weight = self.route_weights[token, rank, None]
            for name in self.outputs:
                storage, mode, register = self.fidelity_cases.get(name, (name, "HiFi4", "SrcA"))
                gate_up, down = (w[offset].float() for w in weights[storage])
                projected = fidelity_linear(x, gate_up, mode, register).bfloat16()
                activated = self.activation(*projected.chunk(2, dim=-1))
                result = fidelity_linear(activated.float(), down, mode, register).bfloat16()
                self.outputs[name].index_add_(0, token, result.float() * routed_weight)

    def finish(self):
        exact = self.outputs["exact"].bfloat16().float()
        normalizer = squared_norm(exact)
        stats = {
            "tokens": self.inputs.shape[0],
            "top_k": self.experts.shape[1],
            "experts_hit": len(self.routes),
            "router_decisions_changed": 0,  # One FP32 router and one set of inputs across all options.
            "relative_l2": {
                name: math.sqrt(squared_norm(value.bfloat16().float() - exact) / normalizer)
                for name, value in self.outputs.items()
                if name != "exact"
            },
        }
        if self.eager is not None:
            hf = self.eager(self.inputs[None])[0]
            simulated = self.outputs["bf16"].bfloat16()
            stats["eager_check"] = {
                "relative_l2": math.sqrt(squared_norm(hf.float() - simulated.float()) / squared_norm(hf)),
                "different_elements": int(torch.count_nonzero(hf != simulated)),
                "total_elements": hf.numel(),
            }
        return stats


def run_layer(shard, layer, config, chunk, output_modes, loss_only, quantizer, loader_only):
    results = {
        "layer": layer,
        "checkpoint_format": "NVFP4" if layer < 20 else "MXFP8",
        "weights": {},
    }
    verify = layer in HF_CHECK_LAYERS and not loss_only
    storage_options = {} if loader_only else STORAGE_OPTIONS
    probe = None if loss_only else MoeProbe(shard, layer, config, verify, output_modes, storage_options)
    references = (
        {name: hf_reference(shard, layer, name, loader_rounding=loader_only) for name in PROJECTIONS} if verify else {}
    )
    for projection in PROJECTIONS:
        results["weights"][projection] = {
            "losses": {},
            "hf_bitwise_equal": verify,
            "squared_norm": 0.0,
            "storage": {}
            if loss_only
            else {name: {"squared_error": 0.0, "max_abs": 0.0, "changed": 0} for name in ("bf16", *storage_options)},
        }

    for start in range(0, 128, chunk):
        experts = slice(start, min(start + chunk, 128))
        variants = {name: [] for name in ("exact", "bf16", *storage_options)}
        for projection in PROJECTIONS:
            decoded = decode(read_raw(shard, layer, projection, experts), projection, loader_rounding=loader_only)
            ref, bf16 = decoded.fp32, decoded.bf16
            width = 2304 if projection == "gate_up_proj" else 1152
            assert ref.shape == (experts.stop - start, 2304, width)
            if verify:
                assert torch.equal(bf16.view(torch.int16), references[projection][experts].view(torch.int16))
            proj = results["weights"][projection]
            for key, value in decoded.losses.items():
                proj["losses"][key] = proj["losses"].get(key, 0) + value
            if loss_only:
                continue

            proj["squared_norm"] += squared_norm(ref)
            update_error(proj["storage"]["bf16"], ref, bf16.float())
            variants["exact"].append(ref)
            variants["bf16"].append(bf16)
            for name, (bits, axis) in storage_options.items():
                weight = quantizer(ref, bits, axis)
                update_error(proj["storage"][name], ref, weight)
                variants[name].append(weight)
        if probe is not None:
            probe.add(start, variants)
        if experts.stop % 32 == 0 and experts.stop < 128 and not loss_only:
            print(f"  {experts.stop}/128 experts", flush=True)

    for proj in results["weights"].values():
        for stats in proj["storage"].values():
            stats["relative_l2"] = math.sqrt(stats["squared_error"] / proj["squared_norm"])
            stats["changed_fraction"] = stats["changed"] / proj["losses"]["elements"]
    if probe is not None:
        results["moe"] = probe.finish()
    return results


def main():
    import transformers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--layers", type=int, nargs="+", choices=range(24), default=LAYERS)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chunk", type=int, choices=range(1, 9), default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--loss-only", action="store_true")
    mode.add_argument(
        "--hf-loader-only", action="store_true", help="Use generic HF BF16 materialization and skip BFP variants"
    )
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument("--fast-bfp", type=Path, help="Shared library built by verify_bfp.py")
    args = parser.parse_args()
    limit_torch(args.threads)

    quantizer = NativeRowQuantizer(args.fast_bfp) if args.fast_bfp else quantize
    config = None
    if not args.loss_only:
        from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

        config = Inkling2LoopTextConfig.from_dict(
            json.loads((args.checkpoint / "config.json").read_text())["text_config"]
        )
        config._experts_implementation = "eager"

    output = {
        "bf16_profile": "generic HF loader" if args.hf_loader_only else "stored companions",
        "packer": "pinned SIMD" if args.fast_bfp else "PyTorch emulator",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "threads": args.threads,
        "chunk": args.chunk,
        "seed": SEED,
        "layers": [],
    }
    with torch.inference_mode(), safe_open(args.checkpoint / SHARD, framework="pt", device="cpu") as shard:
        for layer in args.layers:
            start = time.monotonic()
            print(f"Layer {layer} starting", flush=True)
            result = run_layer(
                shard, layer, config, args.chunk, ("LoFi", "HiFi2"), args.loss_only, quantizer, args.hf_loader_only
            )
            result["seconds"] = time.monotonic() - start
            output["layers"].append(result)
            output["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(output, indent=2) + "\n")
            temporary.replace(args.output)
            print(
                f"Layer {layer} finished in {result['seconds']:.1f}s, "
                f"peak RSS {output['peak_rss_kib'] / 1024**2:.2f} GiB",
                flush=True,
            )


if __name__ == "__main__":
    main()
