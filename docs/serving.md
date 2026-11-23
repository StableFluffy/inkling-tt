# Serving

`serve/openai_server.py` is an OpenAI-compatible HTTP server for one card. It serves one request at a time; the model, its caches and the decode trace stay resident between requests. Run it from the container image or from a source checkout.

The server has no authentication and no rate limiting. It binds to `127.0.0.1` by default. Expose it beyond the host only behind a firewall or an authenticating proxy.

## Container image

You need the converted shards and the tokenizer snapshot from [install.md](install.md) steps 0 and 3, and a free card.

```bash
docker run -d --name inkling --privileged \
  --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G \
  -e TT_VISIBLE_DEVICES=0 -e TT_METAL_VISIBLE_DEVICES=0 \
  -p 127.0.0.1:8000:8000 \
  -v "$INKLING_HF_SD":/models/hf_sd:ro \
  -v "$INKLING_SNAPSHOT":/models/Inkling-2-Mini:ro \
  -v "$INKLING_WORK/image-cache":/cache \
  ghcr.io/stablefluffy/inkling-tt:0.1.0

until curl -sf localhost:8000/health; do sleep 10; done
```

The first start compiles every kernel into `/cache` and takes about four minutes; later starts take about a minute. Arguments after the image name go to the server, for example `--lmax 8192`. The image is about 2.4 GB to pull and 12 GB on disk.

## From a source checkout

```bash
PORT=8000 scripts/serve.sh
```

`HOST=0.0.0.0` makes `serve.sh` listen on every interface of the host, because the dev container shares the host network.

## Requests

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is 2+3?"}],
  "temperature": 0, "max_tokens": 200}'
```

The server implements `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/health`. It supports streaming (SSE), `temperature`, `top_p`, `top_k`, presence and frequency penalties, stop strings and stop token IDs, `seed` and `max_tokens`. `temperature: 0` selects the highest logit after applying any penalties. Without `max_tokens`, a request may generate until the context is full, so set it.

Chat responses put the model's thinking channel in `reasoning_content` and its answer in `content`. `reasoning_effort` is passed to the chat template: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, or a number in [0, 0.99]; the default is 0.9.

Every response carries a `timings` object with `prefill_ms`, `decode_ms`, `ttft_ms` and the derived rates. These are measured inside the server: TTFT covers the request reset and prefill, not queueing, tokenization or network time. With streaming, `timings` arrives in the final usage event.

`tools/bench_server.py` sends a fixed set of short, long and sampled requests and prints their timings:

```bash
INKLING_URL=http://localhost:8000 python3 tools/bench_server.py
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `--lmax` | 32768 | Context length (prompt plus output), multiple of 32 |
| `--prefill-chunk` | 2048, 128 with `--mtp` | Prefill bucket rows; `0` feeds prompts token by token |
| `--host`, `--port` | 127.0.0.1, 8400 | Listen address (the image uses 0.0.0.0:8000 inside the container) |
| `--device-id` | 0 | Card index within the visible devices |
| `--no-trace` | off | Run decode eagerly, for debugging |
| `--mtp [K]` | off | Verify DSpark proposals on greedy requests |

Environment switches for comparisons: `INK_ATTN_DECODE=0` uses TTNN's SDPA decode for attention. `INKLING_EXPERTS=stock` uses TTNN's `sparse_matmul` for experts; because visit fusion depends on the expert streamer, it also turns off the fused attention, router, head tail and token feedback, and decode is much slower.

## How a request runs

The runtime (`serve/runtime.py`) resets the convolution histories, which takes about 5 ms; cache slots past the current position are masked, so the KV caches need no clearing. It splits the prompt into chunks for the two prefill buckets (128 and 2048 rows, see [prefill.md](prefill.md)) and then replays the decode trace once per output token. Greedy requests without penalties take the token from the device argmax and skip the logits readback. The host samples the first token from the prefill logits. The traced decode sampler reads the BF16 head row on the device and applies the FP32 softcap, penalties, temperature and filters across all 200,058 tokens. The host reads a token ID for each subsequent step.

The sampler uses the request's Torch generator for a single uniform draw per token. It walks its cumulative mass in ascending token-ID order and breaks score ties by token ID. Repeating a seeded request reproduces its sequence on the same sampling path. Filtered sequences can differ between the host and device paths because their cumulative order differs. `INK_DEVICE_SAMPLING=0` selects the host path for comparison.

A request may use at most `lmax` tokens in total and must leave room for one output token.

## Speculative decoding

A source checkout can load the official `mtp.safetensors` from `INKLING_SNAPSHOT` and enable
DSpark for greedy requests without penalties. With `--mtp` alone, the server drafts three
proposals per round. Set `"mtp": false` in a chat or completion request to select B1 for that
request. The server uses B1 by default because the speedup depends on the prompt.

```bash
hf download thinkingmachines/Inkling-2-Mini --revision b85c3cfa615fc413cc9af3379f491b963f3b586a \
  --include mtp.safetensors --local-dir "$INKLING_SNAPSHOT"
PORT=8000 scripts/serve.sh --mtp 3 --lmax 8192
INKLING_URL=http://localhost:8000 python3 tools/bench_server.py --compare-mtp
```

The speculative configuration defaults to the 128-row prefill bucket and retains fewer global
attention scores in L1. With the speculative workspaces resident, the 2,048-row expert lane
program exceeded L1 on the tested card. The smaller bucket increases time to first token on
long prompts. See [speculative.md](speculative.md) for the round and measurements.

## vLLM

`serve/vllm_inkling/` holds an experimental adapter for the vLLM TT plugin. It was verified on an earlier build with one sequence and a 2,048-token context and has not been re-run on this release.

```bash
INKLING_DEVICE=none scripts/dev_run.sh "bash serve/install_vllm_venv.sh"
scripts/dev_run.sh "bash serve/vllm_serve.sh"     # listens on 127.0.0.1:8401
```

The tt-metal environment is Python 3.10, so the installer pins vLLM 0.31.0 and the plugin's `compat/vllm-0.31.0` branch (commit `0e96e9ee`) and checks both. The adapter keeps all KV and convolution state inside the model and returns host logits; prefix caching and async decode are off.
