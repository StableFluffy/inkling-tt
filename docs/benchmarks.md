# Benchmarks

Measurements for v0.1.1 on one p150a in the [tested configuration](hardware.md#tested-configuration): 13x10 workers, 380 W, tt-metal `79951bd4`. Everything is batch 1 with the default precision policy (BF16 dense, attention and KV cache, BFP8_B experts, HiFi4 with FP32 accumulation).

## Decode

`scripts/bench_decode.sh` prefills a 29-token prompt and times 60 decode steps per path at `lmax` 8192, after four warm-up steps.

| Path | Median | p95 | Tokens/s |
|---|---:|---:|---:|
| Greedy, device argmax | 18.21 ms | 18.23 ms | 54.9 |
| Full logits read back to the host | 19.11 ms | 19.19 ms | 52.3 |

The v0.1.0 release measured 20.02 ms median and 20.09 ms p95 with device argmax.
The full-grid v0.1.1 trace combines each visit's paired K/V projection and head preparation
and uses a larger kernel-config ring. See [design.md](design.md#where-the-time-goes).

## Prefill

`scripts/run_tests.sh buckets` measures warm prefill from position 0 with both buckets (128 and 2048 rows) resident at `lmax` 32768. Each length uses five warmed runs with the final logits read back to the host.

| Prompt tokens | Median TTFT | p95 TTFT | Tokens/s |
|---:|---:|---:|---:|
| 128 | 0.1694 s | 0.1694 s | 756 |
| 300 | 0.4631 s | 0.4643 s | 648 |
| 1,024 | 0.7097 s | 0.7104 s | 1,443 |
| 2,048 | 0.8919 s | 0.8940 s | 2,296 |
| 8,192 | 3.7450 s | 3.7466 s | 2,187 |
| 32,768 | 18.3952 s | 18.4056 s | 1,781 |

Each chunk goes to whichever bucket costs less for the tokens that remain. A 300-token prompt runs as three 128-row chunks and an 8,192-token prompt runs as four 2048-row chunks. Global attention sees a longer prefix on later chunks. The 32,768-token prompt fills the cache.

## Server

`tools/bench_server.py` runs against a warmed server in the dev image over loopback. Each case uses five non-streaming requests with up to 128 output tokens each. Server TTFT includes request reset and prefill. The decode column reports the median and p95 of per-token decode time across requests.

| Request | Prompt tokens | TTFT median / p95 | Decode median / p95 |
|---|---:|---:|---:|
| Short, greedy | 22 | 0.127 / 0.129 s | 18.52 / 18.64 ms/token |
| Short, greedy | 25 | 0.130 / 0.136 s | 18.47 / 18.65 ms/token |
| Long, greedy | 1,580 | 0.845 / 0.849 s | 19.06 / 19.29 ms/token |
| Long, greedy | 7,820 | 3.679 / 3.689 s | 19.78 / 20.07 ms/token |
| Short, sampled (T=0.8) | 23 | 0.126 / 0.126 s | 19.27 / 19.54 ms/token |

The sampled fixture uses the on-device decode sampler. At a fixed seed, both unfiltered
sampling and sampling with top-k, top-p and penalties reproduced their streams across
three requests.

## Adaptive greedy MTP

`tools/bench_server.py --compare-mtp --runs 3 --max-tokens 64` alternates B1 and adaptive
DSpark requests on a server started with `--mtp --lmax 8192`. A standalone B1 server then
repeats the same seven prompts three times on the same card at `lmax` 8192. All 63 requests
match in streamed choices and usage. Decode and TTFT are in ms, with decode divided by the
number of output IDs after the first. Each cell gives median/p95.

| Prompt | Standalone B1 decode | Resident B1 decode | MTP decode | Standalone B1 TTFT | Resident B1 TTFT | MTP TTFT |
|---|---:|---:|---:|---:|---:|---:|
| Chat | 18.60/18.69 | 18.33/18.35 | 18.45/18.47 | 126.90/132.93 | 121.10/122.54 | 121.90/121.99 |
| Python | 18.41/18.54 | 18.43/18.47 | 16.12/16.26 | 137.70/138.78 | 132.60/132.69 | 132.40/132.58 |
| Korean | 18.42/18.42 | 18.51/18.69 | 18.42/18.56 | 146.20/146.20 | 141.10/141.10 | 141.20/141.56 |
| 1,580 tokens | 19.00/19.10 | 19.42/19.51 | 18.10/18.11 | 846.30/848.28 | 953.50/969.88 | 950.50/957.25 |
| 7,820 tokens | 19.81/19.88 | 20.02/20.16 | 20.10/20.23 | 3664.70/3664.79 | 4332.90/4334.70 | 4335.70/4367.65 |
| Long Python | 18.63/18.70 | 18.71/18.81 | 17.87/17.87 | 282.20/282.65 | 276.70/311.98 | 276.70/276.70 |
| Math | 18.55/18.61 | 18.49/18.62 | 15.78/15.80 | 157.80/157.89 | 153.00/153.18 | 153.20/153.38 |

The mean decode median is 18.78 ms/ID for standalone B1, 18.84 for resident B1 and 17.83 for
adaptive MTP. The 7,820-token prompt takes 0.29 ms/ID longer with MTP than with standalone B1
and adds 0.67 s to TTFT. With MTP resident, the wide prefill bucket keeps fewer scores in L1.
Standalone B1 is the server default.

## Accuracy

These checks compare the port with Hugging Face's eager BF16 implementation on local CPU reference fixtures.

`scripts/run_tests.sh eval` feeds three chat prompts with 128 HF greedy tokens each through the port, teacher-forced, and compares every next-token prediction.

| Metric | Result | Threshold |
|---|---:|---:|
| HF top-1 token is the port's top-1 | 380/384 | 0.97 |
| HF top-1 token in the port's top-5 | 384/384 | 0.99 |
| Mean relative L2 error of the logits | 0.0311 | 0.05 |

`scripts/run_tests.sh parity` checks eager and traced decode against per-step HF captures (9/9 tokens agree) and a 300-token prompt through chunked prefill: relative L2 error of the last logits 0.0289, and 15 of the 16 predictions at and after the prompt end agree with HF.

## Memory and conversion

At `lmax` 32768 with the decode trace, both prefill buckets and the embedding table resident, tt-metal reports 31.36 GB of DRAM allocated out of 33.78 GB.

The converter took 2 minutes 17 seconds, peaked at 59.2 GiB of RAM and wrote 52.8 GB. A repeat conversion matched all 27 reference hashes.

## Reproducing

```bash
scripts/bench_decode.sh
PREFILL_REPS=5 scripts/run_tests.sh buckets
scripts/run_tests.sh eval
scripts/run_tests.sh parity
scripts/dev_run.sh "TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=\$INKLING_WORK/prof python -u tests/prof_trace.py 24"
for run in {1..5}; do INKLING_URL=http://localhost:8000 python3 tools/bench_server.py; done
# Start the MTP server with --mtp --lmax 8192 first.
INKLING_URL=http://localhost:8000 python3 tools/bench_server.py --compare-mtp --runs 3 --max-tokens 64
```

Host scheduling affects per-step wall-time tails. Alternate configurations in one job to compare decode changes.
