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
DSpark requests against the same source server. Both prefill buckets and the standard dispatch
ring are resident with `--mtp 3 --lmax 8192 --prefill-chunk 2048`. All 42 requests match in streamed
choices and usage. Values are median/p95.

| Prompt | B1 decode, ms/ID | MTP decode, ms/ID | B1 TTFT, ms | MTP TTFT, ms |
|---|---:|---:|---:|---:|
| Chat | 19.84/20.07 | 20.02/20.34 | 138.70/149.68 | 141.10/144.70 |
| Python | 19.77/19.78 | 17.27/17.31 | 148.70/150.41 | 148.90/149.62 |
| Korean | 19.78/19.92 | 19.69/19.72 | 158.40/158.76 | 158.50/159.13 |
| 1,580 tokens | 20.51/20.58 | 19.21/19.23 | 1043.60/1049.00 | 1045.80/1046.43 |
| 7,820 tokens | 20.86/21.48 | 21.36/21.86 | 4693.30/4696.36 | 4741.70/4751.87 |
| Long Python | 20.20/20.46 | 19.05/19.22 | 307.10/309.71 | 307.80/309.69 |
| Math | 20.14/20.29 | 15.53/15.58 | 171.50/174.47 | 168.30/174.60 |

The mean of the seven decode medians is 20.16 ms/ID in B1 mode and 18.88 ms/ID with MTP.
The 7,820-token case takes 0.50 ms/ID longer with MTP. B1 is the server default.

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
