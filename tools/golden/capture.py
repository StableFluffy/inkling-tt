# Capture per-layer-application inputs/outputs of HF Inkling2Loop for module-level TT tests.
import collections
import os

import torch
from transformers import AutoTokenizer
from transformers import Inkling2LoopForConditionalGeneration as AM

torch.set_num_threads(32)
P = os.environ["INKLING_SNAPSHOT"]
OUT = os.environ.get("INKLING_GOLDEN", "golden")
os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(P)
m = AM.from_pretrained(P, dtype=torch.bfloat16, device_map="cpu").eval()
lm = m.model.language_model
rec = collections.defaultdict(list)


def pre(name):
    def h(mod, args, kwargs):
        d = {"args": [a.detach().clone() if torch.is_tensor(a) else a for a in args]}
        for k in ("block_states", "messages", "hidden_states", "position_ids", "cache_offset"):
            if k in kwargs:
                v = kwargs[k]
                d[k] = v.detach().clone() if torch.is_tensor(v) else v
        rec[name].append(d)

    return h


def post(name):
    def h(mod, args, kwargs, out):
        rec[name][-1]["out"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

    return h


for i, l in enumerate(lm.layers):
    l.register_forward_pre_hook(pre(f"layer{i}"), with_kwargs=True)
    l.register_forward_hook(post(f"layer{i}"), with_kwargs=True)
for n in ["message_input", "final_input", "norm", "embed_tokens"]:
    mod = getattr(lm, n)
    mod.register_forward_pre_hook(pre(n), with_kwargs=True)
    mod.register_forward_hook(post(n), with_kwargs=True)
msgs = [{"role": "user", "content": "Write a haiku about Tenstorrent chips and explain it briefly."}]
ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")[
    "input_ids"
]
print("prompt", ids.shape, flush=True)
steps = []
with torch.no_grad():
    o = m(input_ids=ids, use_cache=True)
    cache = o.cache
    lg = [o.logits[0, -1].float()]
    steps.append(dict(rec))
    rec.clear()
    gen = []
    for _ in range(8):
        nxt = int(lg[-1].argmax())
        gen.append(nxt)
        o = m(input_ids=torch.tensor([[nxt]]), cache=cache, use_cache=True)
        cache = o.cache
        lg.append(o.logits[0, -1].float())
        steps.append(dict(rec))
        rec.clear()
torch.save({"ids": ids, "gen": gen, "logits": torch.stack(lg), "steps": steps}, os.path.join(OUT, "capture_haiku.pt"))
print("saved", tok.decode(gen), flush=True)
