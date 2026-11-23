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
prompts = [
    "Explain how a looped transformer differs from a standard transformer, in three short paragraphs.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, then explain its complexity.",
    "Tenstorrent의 Blackhole 칩과 GPU의 차이를 한국어로 간단히 설명해 주세요.",
]
out = []
for pr in prompts:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": pr}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"]
    with torch.no_grad():
        o = m(input_ids=ids, use_cache=True)
        cache = o.cache
        lg = [o.logits[0, -1].float()]
        gen = []
        for _ in range(128):
            nxt = int(lg[-1].argmax())
            gen.append(nxt)
            if nxt == 200006:
                break
            o = m(input_ids=torch.tensor([[nxt]]), cache=cache, use_cache=True)
            cache = o.cache
            lg.append(o.logits[0, -1].float())
    out.append({"prompt": pr, "ids": ids[0], "gen": gen, "logits": torch.stack(lg).half()})
    print(len(gen), repr(tok.decode(gen)[:300]), flush=True)
torch.save(out, os.path.join(OUT, "eval3x128.pt"))
print("saved")
