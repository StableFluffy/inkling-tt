import argparse
import asyncio
import time

import torch

from inkling_tt import paths
from serve.openai_server import EOS, Engine, Job, Sampling


def greedy(n):
    return Sampling(
        max_tokens=n,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stop=[],
        stop_ids={EOS},
        seed=0,
    )


def run(engine, ids, n):
    loop = asyncio.new_event_loop()
    job = Job(ids=ids, sampling=greedy(n), out=asyncio.Queue(), loop=loop)
    engine.run(job)
    loop.run_until_complete(asyncio.sleep(0))
    toks = []
    while not job.out.empty():
        toks.append(job.out.get_nowait()[1])
    loop.close()
    return toks, job.timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=paths.hf_sd())
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--lmax", type=int, default=2048)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--no-fast-prefill", action="store_true")
    ap.add_argument("--prefill-chunk", type=int, default=128)
    args = ap.parse_args()
    cap = torch.load(paths.golden("capture_haiku.pt"), weights_only=False)
    ids, ref = cap["ids"][0].tolist(), cap["gen"]
    engine = Engine(args)
    chunk = engine.rt.chunk
    for mode in ("tokenwise", "chunked", "chunked"):
        engine.rt.chunk = chunk if mode == "chunked" else 0
        toks, t = run(engine, ids, 32)
        print(f"prefill={mode} match_hf8={toks[:8] == ref} {t.as_dict()}", flush=True)
        print("  tokens", toks[:16], flush=True)
    long_ids = (ids * 12)[:300]
    out = {}
    for mode in ("tokenwise", "chunked", "chunked"):
        engine.rt.chunk = chunk if mode == "chunked" else 0
        t0 = time.perf_counter()
        engine.rt.reset()
        lg = engine.rt.feed(long_ids, 0)
        dt = (time.perf_counter() - t0) * 1000
        toks, t = run(engine, long_ids, 16)
        out.setdefault(mode, []).append((lg, toks))
        print(f"long {len(long_ids)} tok {mode}: prefill {dt:.0f} ms, run {t.as_dict()}", flush=True)
        print("  tokens", toks, flush=True)
    a, b = out["tokenwise"][0][0], out["chunked"][-1][0]
    print(
        f"long prefill chunked vs tokenwise relerr {((a - b).norm() / a.norm()).item():.3e} "
        f"argmax {int(a.argmax())} {int(b.argmax())} "
        f"greedy16 equal {out['tokenwise'][0][1] == out['chunked'][-1][1]} "
        f"chunked repeat equal {out['chunked'][0][1] == out['chunked'][1][1]}",
        flush=True,
    )
    engine.close()


main()
