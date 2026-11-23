# Benchmarks

Measurements on one p150a in the [tested configuration](hardware.md#tested-configuration): 13x10 workers, 380 W, tt-metal `79951bd4`. Everything is batch 1 with the default precision policy (BF16 dense, attention and KV cache, BFP8_B experts, HiFi4 with FP32 accumulation).

## Decode

`scripts/bench_decode.sh` prefills a 29-token prompt and times 60 decode steps per path at `lmax` 8192, after four warm-up steps.

| Path | Median | p95 | Tokens/s |
|---|---:|---:|---:|
| Greedy, device argmax | 18.37 ms | 18.46 ms | 54.4 |
| Full logits read back to the host | 19.16 ms | 19.36 ms | 52.2 |

In an interleaved comparison with separate K/V projection and head preparation, fusing the two cut the short-prompt greedy median from 18.57 to 18.39 ms. The host-timed p95 moved from 18.82 to 19.19 ms. At 2,048 prompt tokens the greedy median moved from 19.13 to 19.08 ms and p95 from 19.46 to 19.62 ms. Each arm timed 120 decode steps per prompt and preserved all greedy IDs and full logits.

## Prefill

`scripts/run_tests.sh buckets` measures warm prefill from position 0 with both buckets (128 and 2048 rows) resident at `lmax` 32768. Each length uses five warmed runs with the final logits read back to the host.

| Prompt tokens | Median TTFT | p95 TTFT | Tokens/s |
|---:|---:|---:|---:|
| 22 | 0.1202 s | 0.1211 s | 183 |
| 128 | 0.1710 s | 0.2072 s | 749 |
| 300 | 0.4648 s | 0.4692 s | 645 |
| 1,024 | 0.7206 s | 0.7339 s | 1,421 |
| 2,048 | 0.9119 s | 0.9620 s | 2,246 |
| 8,192 | 3.7759 s | 3.7827 s | 2,170 |
| 32,768 | 18.4894 s | 18.4899 s | 1,772 |

Packing the repeat-pass message queries and consuming BF16 controller inputs directly reduced the 2,048-token median from 0.918 to 0.899 s in a paired comparison. The 8,192-token median went from 3.820 to 3.772 s, with bitwise-equal logits.

Each chunk goes to whichever bucket costs less for the tokens that remain. A 300-token prompt runs as three 128-row chunks and an 8,192-token prompt runs as four 2048-row chunks. Global attention sees a longer prefix on later chunks. The 32,768-token prompt fills the cache.

## Server

`tools/bench_server.py` runs against a warmed server in the dev image over loopback. Each case uses five non-streaming requests with up to 128 output tokens each. Server TTFT includes request reset and prefill. The decode column reports the median and p95 of per-token decode time across requests.

| Request | Prompt tokens | TTFT median / p95 | Decode median / p95 |
|---|---:|---:|---:|
| Short, greedy | 22 | 0.127 / 0.127 s | 18.7 / 18.9 ms/token |
| Short, greedy | 25 | 0.131 / 0.135 s | 18.7 / 19.0 ms/token |
| Long, greedy | 1,580 | 0.852 / 0.856 s | 19.4 / 19.5 ms/token |
| Long, greedy | 7,820 | 3.709 / 3.714 s | 19.8 / 20.0 ms/token |
| Short, sampled (T=0.8) | 23 | 0.126 / 0.146 s | 19.5 / 19.9 ms/token |

## Adaptive greedy MTP

`tools/bench_server.py --compare-mtp --runs 3 --max-tokens 64` alternates B1 and adaptive
DSpark requests on a server started with `--mtp 3 --lmax 8192 --prefill-chunk 2048`.
The same job then starts a standalone B1 server and repeats all seven prompts three times.
Both servers use the larger dispatch ring. All 63 requests match in streamed choices and usage.
Decode is in ms/ID and TTFT is in ms. Each cell gives median/p95.

| Prompt | Standalone B1 decode | Resident B1 decode | MTP decode | Standalone B1 TTFT | Resident B1 TTFT | MTP TTFT |
|---|---:|---:|---:|---:|---:|---:|
| Chat | 18.46/18.46 | 18.55/18.57 | 18.54/18.65 | 128.90/134.75 | 123.50/123.77 | 123.70/124.15 |
| Python | 18.44/18.45 | 18.63/18.64 | 16.19/16.21 | 139.10/139.19 | 134.70/134.97 | 134.50/134.77 |
| Korean | 18.44/18.46 | 18.94/19.04 | 18.75/18.82 | 147.90/148.53 | 144.50/146.12 | 143.60/143.78 |
| 1,580 tokens | 19.02/19.04 | 19.41/19.52 | 18.21/18.21 | 858.90/860.16 | 968.00/968.90 | 961.00/979.36 |
| 7,820 tokens | 19.91/20.19 | 20.35/20.37 | 20.28/20.38 | 3711.60/3713.58 | 4372.00/4376.59 | 4367.90/4374.20 |
| Long Python | 18.99/19.09 | 18.88/18.88 | 17.87/17.89 | 287.60/306.86 | 280.20/280.83 | 280.40/280.94 |
| Math | 18.78/19.01 | 18.77/18.83 | 15.80/15.81 | 160.70/162.86 | 155.20/155.20 | 154.60/154.78 |

The mean decode median is 18.86 ms/ID for standalone B1, 19.08 for resident B1 and 17.95
for adaptive MTP. At 7,820 tokens the MTP median is 0.37 ms/ID higher and TTFT is 656 ms higher
than standalone B1. B1 is the server default.

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

At `lmax` 32768 with the decode trace, both prefill buckets and the embedding table resident, tt-metal reports 31.35 GB of DRAM allocated out of 33.78 GB.

The converter took 2 minutes 17 seconds, peaked at 59.2 GiB of RAM and wrote 52.8 GB. A repeat conversion matched all 27 reference hashes.

## Reproducing

```bash
scripts/bench_decode.sh
PREFILL_REPS=5 PREFILL_LENGTHS=22,128,300,1024,2048,8192,32768 scripts/run_tests.sh buckets
scripts/run_tests.sh eval
scripts/run_tests.sh parity
scripts/dev_run.sh "TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=\$INKLING_WORK/prof python -u tests/prof_trace.py 24"
INKLING_URL=http://localhost:8000 python3 tools/bench_server.py
```

Host scheduling affects per-step wall-time tails. Alternate configurations in one job to compare decode changes.
