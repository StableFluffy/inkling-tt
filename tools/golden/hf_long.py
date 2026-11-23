import os

import torch
from transformers import AutoTokenizer
from transformers import Inkling2LoopForConditionalGeneration as AM

torch.set_num_threads(24)
P = os.environ["INKLING_SNAPSHOT"]
OUT = os.environ.get("INKLING_GOLDEN", "golden")
os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(P)
m = AM.from_pretrained(P, dtype=torch.bfloat16, device_map="cpu").eval()
with open(os.path.join(P, "README.md")) as f:
    text = f.read()[3000:7000]
msgs = [{"role": "user", "content": "Summarize the following model card excerpt in two sentences:\n\n" + text}]
ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")[
    "input_ids"
][:, :300]
with torch.no_grad():
    o = m(input_ids=ids, use_cache=True)
    cache = o.cache
    lg = [o.logits[0, -1].float()]
    gen = []
    for _ in range(16):
        nxt = int(lg[-1].argmax())
        gen.append(nxt)
        o = m(input_ids=torch.tensor([[nxt]]), cache=cache, use_cache=True)
        cache = o.cache
        lg.append(o.logits[0, -1].float())
torch.save({"ids": ids[0], "gen": gen, "logits": torch.stack(lg)}, os.path.join(OUT, "long300.pt"))
print("saved", ids.shape, tok.decode(gen))
