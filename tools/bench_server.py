import argparse
import json
import os
import statistics
import time
import urllib.request

URL = os.environ.get("INKLING_URL", "http://localhost:8000")


def req(msg, max_tokens=128, temperature=0.0, *, stream=False, mtp=True):
    body = json.dumps(
        {
            "model": "Inkling-2-Mini-TT",
            "messages": [{"role": "user", "content": msg}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "mtp": mtp,
        }
    ).encode()
    r = urllib.request.Request(URL + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(r) as response:
        if not stream:
            return json.load(response), time.perf_counter() - t0
        choices = []
        final = None
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError(event["error"])
            if "timings" in event:
                final = event
            choices.extend(event["choices"])
        if final is None:
            raise RuntimeError("Completion stream ended without timings")
        final["choices"] = choices
        return final, time.perf_counter() - t0


def report(samples):
    median = statistics.median(samples)
    p95 = statistics.quantiles(samples, n=20, method="inclusive")[-1] if len(samples) > 1 else median
    return f"{median:.2f}/{p95:.2f} ms median/p95"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-mtp", action="store_true")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.runs < 1 or args.max_tokens < 2:
        parser.error("benchmarks need at least one run and two output tokens")
    long = "Tenstorrent Blackhole p150a cards connect over Ethernet. " * 120
    chat = ("chat", "Explain KV caching in two sentences.")
    if args.compare_mtp:
        for name, msg in (
            chat,
            ("code", "Write a Python function to compute the longest common prefix of a list of strings."),
            ("Korean", "Tenstorrent의 Blackhole 칩과 GPU의 차이를 한국어로 간단히 설명해 주세요."),
            ("1k", "Summarize this: " + long),
            ("8k", "Summarize this: " + long * 5),
        ):
            streams = None
            times = {False: [], True: []}
            first = {False: [], True: []}
            for i in range(args.runs):
                for mtp in (False, True) if i % 2 == 0 else (True, False):
                    d, _ = req(msg, args.max_tokens, stream=True, mtp=mtp)
                    output = d["choices"], d["usage"]
                    if streams is None:
                        streams = output
                    assert output == streams, f"Greedy stream changed for {name} with MTP={mtp}"
                    t = d["timings"]
                    times[mtp].append(t["ms_per_token"])
                    first[mtp].append(t["ttft_ms"])
            for mtp in (False, True):
                label = "MTP" if mtp else "B1"
                print(f"{name} {label}: decode {report(times[mtp])}, TTFT {report(first[mtp])}", flush=True)
        return
    for name, msg in (
        ("short", "Write a haiku about Tenstorrent chips."),
        ("short", chat[1]),
        ("1k", "Summarize this: " + long),
        ("8k", "Summarize this: " + long * 5),
        ("sampled", "Tell me a story about a robot."),
    ):
        d, wall = req(msg, args.max_tokens, temperature=(0.8 if name == "sampled" else 0.0))
        u = d.get("usage", {})
        tm = d.get("timings", {})
        print(
            name,
            "prompt",
            u.get("prompt_tokens"),
            "gen",
            u.get("completion_tokens"),
            f"wall {wall:.2f}s",
            json.dumps(tm)[:300],
            flush=True,
        )


if __name__ == "__main__":
    main()
