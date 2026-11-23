"""Compute expert tile payload, model residency and decode traffic from shard headers."""

import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path

from dequant import CHECKPOINT, SHARD

TILE_BYTES = {"bfp4_b": 576, "bfp8_b": 1088, "bf16": 2048}
FORMATS = {
    "native": ("native", "native"),
    "bfp4_b": ("bfp4_b", "bfp4_b"),
    "bfp4_b / bfp8_b": ("bfp4_b", "bfp8_b"),
    "bfp8_b": ("bfp8_b", "bfp8_b"),
    "bfp8_b / bf16": ("bfp8_b", "bf16"),
    "bf16": ("bf16", "bf16"),
}


def read_header(path):
    with path.open("rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        return {key: val for key, val in json.loads(f.read(length)).items() if key != "__metadata__"}


def nbytes(tensor):
    begin, end = tensor["data_offsets"]
    return end - begin


def calculate(checkpoint):
    config = json.loads((checkpoint / "config.json").read_text())["text_config"]
    experts = config["n_routed_experts"]
    top_k = config["num_experts_per_tok"]
    loop = config["loop_config"]
    visits = {
        layer: loop["num_passes"] if loop["core_start"] <= layer < loop["core_end"] else 1
        for layer in range(config["num_hidden_layers"])
    }
    native = defaultdict(int)
    denominator = defaultdict(int)
    tiles = defaultdict(int)
    text = vision = router_expansion = common_traffic = 0
    for key, tensor in read_header(checkpoint / SHARD).items():
        size = nbytes(tensor)
        if ".mlp.experts." in key:
            layer = int(key.split(".")[3])
            native[layer] += size
            if key.endswith("_denominator"):
                denominator[layer] += size
            if key.endswith("_weight"):
                e, n, k = tensor["shape"]
                if tensor["dtype"] == "U8":
                    k *= 2
                assert e == experts and n % 32 == k % 32 == 0
                tiles[layer] += e * n * k // 1024
        elif key.startswith("model.visual."):
            vision += size
        else:
            text += size
            if key.endswith(".mlp.gate.weight") and tensor["dtype"] == "BF16":
                router_expansion += size  # FP32 uses twice its checkpoint bytes.
                size *= 2
            if key.startswith("model.llm.layers."):
                layer = int(key.split(".")[3])
                count = visits[layer] - int(".controllers." in key)
                common_traffic += size * count
            elif key.endswith(".unembed.weight"):
                common_traffic += size // tensor["shape"][0] * config["unpadded_vocab_size"]
            elif key.endswith(".embed.weight"):
                common_traffic += size // tensor["shape"][0]
            elif key in ("model.llm.norm.weight", "model.llm.embed_norm.weight"):
                common_traffic += size
    text += router_expansion
    draft_path = checkpoint / "mtp.safetensors"
    draft = sum(nbytes(tensor) for tensor in read_header(draft_path).values()) if draft_path.exists() else None
    result = {
        "nonexpert_text_bytes": text,
        "router_fp32_expansion_bytes": router_expansion,
        "vision_bytes": vision,
        "draft_bytes": draft,
        "visits_nvfp4": sum(n for layer, n in visits.items() if layer < 20),
        "visits_mxfp8": sum(n for layer, n in visits.items() if layer >= 20),
        "experts_per_visit": top_k,
        "tiles_per_expert": tiles[0] // experts,
        "common_text_decode_bytes_per_token": common_traffic,
        "options": {},
    }

    for name, (first, last) in FORMATS.items():
        resident = traffic = 0
        for layer, count in visits.items():
            fmt = first if layer < 20 else last
            if fmt == "native":
                size = native[layer]
                shared = denominator[layer]
            else:
                size = TILE_BYTES[fmt] * tiles[layer]
                shared = 0
            assert (size - shared) % experts == 0
            resident += size
            traffic += count * ((size - shared) // experts * top_k + shared)
        target = resident + text + vision
        result["options"][name] = {
            "expert_bytes": resident,
            "text_bytes": resident + text,
            "target_bytes": target,
            "target_with_draft_bytes": target + draft if draft is not None else None,
            "target_one_card_headroom_bytes": 32_000_000_000 - target,
            "target_two_card_even_bytes_per_card": target // 2,
            "target_two_card_ep_bytes_per_card": resident // 2 + text + vision,
            "expert_decode_bytes_per_token": traffic,
            "text_weight_decode_bytes_per_token": traffic + common_traffic,
        }
    local = sum(visits[layer] for layer in config["local_layer_ids"])
    cache_per_slot_token = 2 * config["num_key_value_heads"] * config["head_dim"] * 2
    result["bf16_kv_bytes_b1"] = {
        length: cache_per_slot_token
        * (local * min(length, config["sliding_window_size"]) + (sum(visits.values()) - local) * length)
        for length in (1024, 32768, 131072, 1048576)
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    print(json.dumps(calculate(args.checkpoint), indent=2))
