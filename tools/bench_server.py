import json
import os
import time
import urllib.request

URL = os.environ.get("INKLING_URL", "http://localhost:8000")


def req(msg, max_tokens=128, temperature=0.0, stream=False):
    body = json.dumps(
        {
            "model": "Inkling-2-Mini-TT",
            "messages": [{"role": "user", "content": msg}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()
    r = urllib.request.Request(URL + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(r))
    wall = time.time() - t0
    return d, wall


long = "Tenstorrent Blackhole p150a cards connect over Ethernet. " * 120
for name, msg in (
    ("short", "Write a haiku about Tenstorrent chips."),
    ("short", "Explain KV caching in two sentences."),
    ("1k", "Summarize this: " + long),
    ("8k", "Summarize this: " + long * 5),
    ("sampled", "Tell me a story about a robot."),
):
    d, wall = req(msg, temperature=(0.8 if name == "sampled" else 0.0))
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
