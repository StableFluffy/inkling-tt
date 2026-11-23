import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import safetensors
import torch
import transformers
from safetensors.torch import save_file
from transformers import Inkling2LoopForConditionalGeneration

CONVERTER_VERSION = "1"


def snapshot_revision(src):
    meta = os.path.join(src, ".cache", "huggingface", "download", "config.json.metadata")
    if os.path.exists(meta):
        with open(meta) as f:
            return f.readline().strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if os.listdir(args.out):
        raise SystemExit(f"{args.out} is not empty; convert into a fresh directory")
    sd = Inkling2LoopForConditionalGeneration.from_pretrained(
        args.src, dtype=torch.bfloat16, device_map="cpu"
    ).state_dict()
    lm = "model.language_model."
    groups = {}
    for k, v in sd.items():
        if k.startswith(lm + "layers."):
            i = int(k[len(lm + "layers.") :].split(".")[0])
            groups.setdefault(f"layer{i:02d}", {})[k[len(lm) :]] = v.contiguous()
        elif k.startswith(lm) or k.startswith("lm_head"):
            groups.setdefault("misc", {})[k.replace(lm, "")] = v.contiguous()
        else:
            groups.setdefault("vision", {})[k] = v.contiguous()
    idx = {}
    for g, d in groups.items():
        save_file(d, os.path.join(args.out, f"{g}.safetensors"))
        for k, v in d.items():
            idx[k] = [g, str(v.dtype), list(v.shape)]
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(idx, f, indent=0)
    manifest = {
        "base_model": "thinkingmachines/Inkling-2-Mini",
        "revision": args.revision or snapshot_revision(args.src),
        "converter_version": CONVERTER_VERSION,
        "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "safetensors": safetensors.__version__,
        "dtype": "bfloat16",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "groups": sorted(groups),
    }
    with open(os.path.join(args.out, "convert_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("done", len(idx))


if __name__ == "__main__":
    main()
