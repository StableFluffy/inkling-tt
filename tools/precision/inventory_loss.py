"""Count BF16-inexact native expert weights across all physical layers."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from dequant import CHECKPOINT, E2M1, PROJECTIONS, SHARD, decode, limit_torch, read_raw
from safetensors import safe_open


def count_nvfp4(raw):
    weight, scale, amax, denominator = raw
    assert denominator.numel() == 1 and denominator.item() == 2688
    assert torch.isfinite(scale.float()).all()
    code = torch.arange(256)
    pairs = torch.stack((E2M1[code & 15], E2M1[code >> 4]), dim=-1).double()
    scale_values = code.byte().view(torch.float8_e4m3fn).float().double()
    inner = pairs[None, :, :] * scale_values[:, None, None]
    native = (inner[None] * amax.double()[:, None, None, None]) / denominator.double()
    # The lookup stores 0, 1 or 2 inexact BF16 lanes per packed byte.
    count_table = (native != native.float().bfloat16().double()).sum(-1, dtype=torch.uint8).numpy()
    weight_bytes = weight.view(*scale.shape, 8).numpy()
    scale_bytes = scale.view(torch.uint8).numpy()
    changed = sum(
        int(table[scales[..., None], packed].sum(dtype=np.int64))
        for table, scales, packed in zip(count_table, scale_bytes, weight_bytes)
    )
    return 2 * weight.numel(), changed


def count_mxfp8(raw, projection):
    weight, scale = raw
    assert not np.any((weight.view(torch.uint8).numpy() & 127) == 127)
    # Every nonzero E4M3FN value has an exponent in [-9, 8] and at most
    # four significant bits. This range test proves the scaled results
    # fit in normal BF16, including the E4M3FN subnormals.
    low, high = torch.aminmax(scale)
    if low >= 10 and high <= 246:
        return weight.numel(), 0
    return weight.numel(), decode(raw, projection).losses["native_bf16_loss"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=Path("inventory.json"))
    args = parser.parse_args()
    limit_torch(1)
    result = {}
    with safe_open(args.checkpoint / SHARD, framework="pt", device="cpu") as shard:
        for layer in range(24):
            result[layer] = {}
            for projection in PROJECTIONS:
                elements = changed = 0
                for start in range(0, 128, 4):
                    raw = read_raw(shard, layer, projection, slice(start, start + 4))
                    count, losses = count_nvfp4(raw) if len(raw) == 4 else count_mxfp8(raw, projection)
                    elements += count
                    changed += losses
                result[layer][projection] = {"elements": elements, "native_bf16_loss": changed}
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(f"Layer {layer} complete", flush=True)


if __name__ == "__main__":
    main()
