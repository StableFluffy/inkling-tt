# Development

How to run the checks, measure changes and build the release image. Setup is in [install.md](install.md).

## Environment

The wrappers in `scripts/` read these variables; defaults are in `scripts/env.sh`.

| Variable | Meaning |
|---|---|
| `TT_METAL_HOME` | tt-metal checkout at the pinned commit |
| `TT_METAL_VENV` | Python environment with `ttnn` and the server packages |
| `INKLING_SNAPSHOT` | Official checkpoint snapshot (tokenizer, chat template, config) |
| `INKLING_HF_SD` | Converted per-layer shards |
| `INKLING_GOLDEN` | HF reference captures |
| `INKLING_WORK` | Logs, kernel cache, container home |
| `INKLING_DEVICE` | Card index, or `none` for host-only commands |
| `INKLING_LOCK_DIR` | Optional directory of per-card lock files on shared hosts |

Model switches, read at model construction:

| Variable | Default | Effect |
|---|---|---|
| `INKLING_EXPERTS` | `stream` | `stock`: TTNN `sparse_matmul` experts, without visit fusion |
| `INK_ATTN_DECODE` | `1` | `0` runs decode attention through TTNN SDPA decode |
| `INK_PAIR_KV` | `1` | `0` runs separate K and V projections during decode |
| `INK_GRID` | unset | Restrict scheduling to `11x10` or `12x10` ([hardware.md](hardware.md)) |
| `INK_LMAX` | test-specific | Context length used by the tests and benchmarks |

Each decode fusion also has an `INK_*` switch near the top of `InklingTT2.__init__` (`inkling_tt/model2.py`), on by default, so a change can be compared against the path it replaced.

`scripts/dev_run.sh "<command>"` runs one command inside the pinned dev image with the repository mounted, the selected card visible and the kernel cache in `$INKLING_WORK/kcache`. Every other script goes through it.

## Tests

`scripts/run_tests.sh <tier>` runs one tier and stops at the first failure. Logs go to `$INKLING_WORK/logs/`.

| Tier | Device | Checks |
|---|---|---|
| `host` | no | Sources compile; ttnn, the model and the server import |
| `parity` | yes | Eager and traced decode, chunked prefill and a 300-token prompt against HF |
| `eval` | yes | Teacher-forced top-1, top-5 and logit error over 3 prompts x 128 tokens |
| `buckets` | yes | Both prefill buckets at `lmax` 32768, bucket equality, decode after prefill |
| `kernels` | yes | rowvm, expert-stream and attention kernels against PyTorch |
| `serve` | yes | The serving engine against the HF capture, without HTTP |

`parity` and `eval` assert the thresholds listed in [benchmarks.md](benchmarks.md#accuracy). The `kernels` tier prints error statistics per case for reading by hand.

Run `parity` and `eval` after any change to model code or kernels. Changes that only reorder FP32 sums can move individual logits slightly; bit-preserving changes should reproduce the per-position errors of the previous run exactly, which is the easier thing to check.

## Benchmarks

```bash
scripts/bench_decode.sh                                       # greedy and logits-readback decode, lmax 8192
scripts/dev_run.sh "python -u tests/bench_prefill.py 128 2048" # warm TTFT per prompt length
INKLING_URL=http://localhost:8000 python3 tools/bench_server.py  # end to end through HTTP
```

Decode timings are medians over 64 tokens after four warm-up tokens. Run A/B comparisons in the same process or interleaved in one job; host jitter alone moves a single run by about 0.3 ms.

## Profiling

The tt-metal device profiler gives per-program kernel times for a trace replay:

```bash
scripts/dev_run.sh "TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -p -o \$INKLING_WORK/prof tests/prof_trace.py 24"
```

`tests/prof_trace.py` captures the decode trace for the given number of layers (default 7) and replays three steps under the profiler.

Profiled spans run 0.2-0.4 ms longer than unprofiled replay. Group the rows of `cpp_device_perf_report.csv` by kind of work, as in the table in [design.md](design.md#where-the-time-goes), and compare kernel sums with the firmware span to separate kernel time from gaps. When the weight-streaming rows are close to their byte cost, the remaining budget is the glue and the gaps.

## Trace rules

- Allocate every tensor the step uses before `ttnn.begin_trace_capture`, and run every program once eagerly first. A buffer allocated or a program compiled during capture can share memory with trace intermediates, and the replay then corrupts it or hangs the card.
- Build every `Prefill` instance before `m.capture()`. `serve/runtime.py` does this in the right order.
- Inputs that change per step live in persistent device tensors; write them with `ttnn.copy_host_to_device_tensor` before the replay, never by allocating new tensors.

If a card hangs, stop the process and reset it with `tt-smi -r <PCI address>`. Two cards joined by Ethernet must be reset together.

## Writing a kernel

Custom programs use `ttnn.generic_op` with kernels under `inkling_tt/kernels/`. tt-metal compiles them on first use and caches the binaries, so no tt-metal rebuild is needed. [rowvm.md](rowvm.md) shows how a row epilogue is described and launched; it is the quickest way to add fused element-wise or reduction work between matmuls. Size core grids from `inkling_tt/grid.py`, not from constants, so the program runs on every supported grid.

## Release image

```bash
export TT_METAL_HOME=/path/to/clean/tt-metal
REF=v0.1.0 VERSION=0.1.0 TAG=inkling-2-mini-tt:0.1.0 serve/build_image.sh
```

`build_image.sh` packages `git archive $REF`, so uncommitted files never reach the image, and labels the image with that commit. It adds the built tt-metal tree without its git data, build objects or docs, and refuses a tt-metal tree at another commit, with local changes, or whose files contain the host build path. `scripts/publish_image.sh` checks the labels, the packaged code and the license files again before it pushes.

The base is the tt-metal dev image pinned by digest. The published image contains all of its layers, so it keeps working if the base disappears upstream. To rebuild without the base, rebuild it from tt-metal's `dockerfile/` at the pinned commit, set `INKLING_DEV_IMAGE` to the new digest, and rerun `parity`, `eval` and the container smoke test before publishing a new version.
