# Benchmarks

Release measurements for v0.1.0, taken from a fresh clone of the release commit and a fresh tt-metal build, on one p150a in the [tested configuration](hardware.md#tested-configuration): 13x10 workers, 380 W, tt-metal `79951bd4`. Everything is batch 1 with the default precision policy (BF16 dense, attention and KV cache, BFP8_B experts, HiFi4 with FP32 accumulation).

## Decode

`scripts/bench_decode.sh`: prefill a short prompt, then 64 decode steps with changing tokens and positions; the first four are discarded. `lmax` 8192.

| Path | Median | p95 | Tokens/s |
|---|---:|---:|---:|
| Greedy, device argmax | 20.02 ms | 20.09 ms | 49.9 |
| Full logits read back to the host | 20.75 ms | 20.87 ms | 48.2 |
| Greedy, stock SDPA attention (`INK_ATTN_DECODE=0`) | 22.79 ms | 22.84 ms | 43.9 |
| Greedy, restricted to 11x10 (`INK_GRID=11x10`) | 20.54 ms | 20.58 ms | 48.7 |

Decode slows slightly with context length because global-attention visits read more cache: about 20.7 ms/token after a 1.6k-token prompt and 21.3 ms/token after 7.8k tokens (server measurements below).

## Prefill

`scripts/run_tests.sh buckets`: warm prefill from position 0 with both buckets (128 and 2048 rows) resident at `lmax` 32768, including the readback of the last logits. Median of three runs.

| Prompt tokens | Time to first token | Tokens/s |
|---:|---:|---:|
| 128 | 0.207 s | 618 |
| 300 | 0.567 s | 530 |
| 1,024 | 0.852 s | 1,201 |
| 2,048 | 1.036 s | 1,976 |
| 8,192 | 4.313 s | 1,900 |
| 32,768 | 22.88 s | 1,432 |

Each chunk goes to whichever bucket costs less for the tokens that remain, so a 300-token prompt runs as three 128-row chunks and an 8,192-token prompt as four 2048-row chunks. Throughput falls at 32k because attention over the global layers grows with the prompt. The 32,768-token row fills the whole cache and leaves no room for output.

## Server

`tools/bench_server.py` against the container image, one non-streaming request per case, 128 output tokens at most. TTFT is reported by the server: request reset plus prefill.

| Request | Prompt tokens | TTFT | Decode |
|---|---:|---:|---:|
| Short, greedy | 22 | 0.155 s | 20.1 ms/token |
| Short, greedy | 25 | 0.159 s | 20.2 ms/token |
| Long, greedy | 1,580 | 0.989 s | 20.7 ms/token |
| Long, greedy | 7,820 | 4.239 s | 21.3 ms/token |
| Short, sampled (T=0.8) | 23 | 0.154 s | 21.5 ms/token |

Sampled requests read the logits back and sample on the host. The container needs 217 s to answer `/health` on its first start, which compiles every kernel, and 66 s when the kernel cache is warm.

## Accuracy

The reference is Hugging Face's eager implementation in BF16 on the CPU. These checks measure how closely the port tracks it on local fixtures; they are not a task evaluation.

`scripts/run_tests.sh eval` feeds three chat prompts with 128 HF greedy tokens each through the port, teacher-forced, and compares every next-token prediction.

| Metric | Result | Threshold |
|---|---:|---:|
| HF top-1 token is the port's top-1 | 380/384 | 0.97 |
| HF top-1 token in the port's top-5 | 384/384 | 0.99 |
| Mean relative L2 error of the logits | 0.0311 | 0.05 |

`scripts/run_tests.sh parity` checks eager and traced decode against per-step HF captures (9/9 tokens agree) and a 300-token prompt through chunked prefill: relative L2 error of the last logits 0.0289, and 15 of the 16 predictions at and after the prompt end agree with HF. The same fixtures give bit-identical logits at 11x10, 12x10 and 13x10.

## Memory and conversion

At `lmax` 32768 with the decode trace, both prefill buckets and the embedding table resident, tt-metal reports 31.05 GB of DRAM allocated out of 33.78 GB.

The converter took 2 minutes 17 seconds, peaked at 59.2 GiB of RAM and wrote 52.8 GB. A repeat conversion matched all 27 reference hashes.

## Reproducing

```bash
scripts/bench_decode.sh
INK_ATTN_DECODE=0 scripts/bench_decode.sh
INK_GRID=11x10 scripts/bench_decode.sh
scripts/run_tests.sh buckets
scripts/run_tests.sh eval
scripts/run_tests.sh parity
INKLING_URL=http://localhost:8000 python3 tools/bench_server.py
```

Decode varies by about 0.3 ms between runs because of host jitter. Compare two configurations in the same job, alternating between them.
