from pathlib import Path

import torch
from safetensors.torch import load_file

from inkling_tt.model import HD, HKV, HQ, VOCAB, D

RANK = 256
MLP_WIDTH = 9216
PADDED_VOCAB = 201024


def _checkpoint_shapes():
    shapes = {
        "norm.weight": (D,),
        "mask_embed": (D,),
        "fuse_weight": (4, 4),
        "markov_head.markov_w1.weight": (PADDED_VOCAB, RANK),
        "markov_head.rnn_proj.weight": (3 * RANK, 2 * (D + RANK)),
        "markov_head.rnn_proj.bias": (3 * RANK,),
        "markov_head.hidden_out.weight": (D, RANK),
        "markov_head.hidden_out.bias": (D,),
        "markov_head.dflare_norm.weight": (D,),
    }
    for i in range(3):
        layer = f"layers.{i}."
        for name in ("dflare_norm", "input_layernorm", "post_attention_layernorm"):
            shapes[f"{layer}{name}.weight"] = (D,)
        for name in ("k_norm", "q_norm", "k_norm_ctx"):
            shapes[f"{layer}self_attn.{name}.weight"] = (HD,)
        for name in ("k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx"):
            shapes[f"{layer}self_attn.{name}.weight"] = (HKV * HD, D)
        shapes[f"{layer}self_attn.q_proj.weight"] = (HQ * HD, D)
        shapes[f"{layer}self_attn.o_proj.weight"] = (D, HQ * HD)
        for name in ("gate_proj", "up_proj"):
            shapes[f"{layer}mlp.{name}.weight"] = (MLP_WIDTH, D)
        shapes[f"{layer}mlp.down_proj.weight"] = (D, MLP_WIDTH)
    return shapes


def load_drafter(snapshot):
    weights = load_file(Path(snapshot) / "mtp.safetensors", device="cpu")
    shapes = _checkpoint_shapes()
    if weights.keys() != shapes.keys():
        raise ValueError(f"Unexpected DSpark tensors: {sorted(weights.keys() ^ shapes.keys())}")
    for name, shape in shapes.items():
        tensor = weights[name]
        if tensor.shape != shape or tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name}: expected BF16 {shape}, got {tensor.dtype} {tuple(tensor.shape)}")
    return weights


def fold_head(head, weights):
    if head.shape != (VOCAB, D) or head.dtype != torch.bfloat16 or head.device.type != "cpu":
        raise ValueError("Expected the CPU BF16 target LM head with the real vocabulary")
    latent = weights["markov_head.hidden_out.weight"].double()
    bias = weights["markov_head.hidden_out.bias"].double()
    folded = head.new_empty((VOCAB, RANK))
    folded_bias = head.new_empty(VOCAB)
    for start in range(0, VOCAB, 4096):
        rows = slice(start, start + 4096)
        block = head[rows].double()
        folded[rows] = block @ latent
        folded_bias[rows] = block @ bias
    return folded, folded_bias
