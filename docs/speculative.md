# DSpark draft and verification kernels

The Inkling-2-Mini checkpoint ships a three-layer DSpark drafter in `mtp.safetensors`. The device
drafter uses target features to propose several tokens for one pending token. A lossless round
requires the target to reproduce B1 logits and state for each proposed input. Only the matching
prefix advances the target position. The verifier processes each visit in layer order across the block. It
shares the attention projections, LM head and routed-expert weights across the verify rows.

## Draft context

The four BF16 inputs are the incoming partials at chronological visits 5, 23 and 41 and the final
normalized state. `InklingTT2` and `Prefill` can copy these rows into persistent device buffers.
The first three taps precede physical layers 5, 11 and 17 on passes 0, 1 and 2 respectively. Prefill
captures the complete chunk before selecting the last row for the target head. Its descriptor
identifies the rows to append to the drafter's context.

Each draft layer forms its own FP32 mixture of the taps with checkpoint fusion weights, then applies
the layer's weighted RMSNorm. The layer projects BF16 K and V from the mixture and applies a
learned K norm and NeoX RoPE to K. The first and third layers keep 1,024-slot rings. The middle
layer keeps the full prefix. When the maximum context exceeds 2,048, the round also keeps a
1,024-slot copy of the global draft cache for short prefixes. The long draft uses a tiled FP32
softmax whose L1 scratch fits alongside the target and prefill workspaces. A fourth tap mixture supplies the Markov context from the last
committed input.

## Draft scores

The draft block starts at the first uncommitted position. It embeds the pending target ID in row 0
and fills subsequent rows with the checkpoint mask embedding. Each query attends to its layer's
committed context and the entire transient block. One RoPE program rotates each batch of
heads at their absolute positions. It keeps the products and sums in FP32 and rounds to BF16. The
transient self-projected K/V occupy separate slots from the feature-projected prefix. The
three query heads in each KV group pack their live rows into one tile. This cuts score and
probability storage to one-third for the draft's attention products.

Rows 1 through K predict the K proposals at their own positions. The block reads the target
unembedding once for their BF16 base scores. A copy from the width-sharded head output collects
only each proposal's live row. The Markov head starts each round with a zero
recurrent state and uses the previous ID, the draft hidden row and the last context mixture. Its
256-wide output projects through the checkpoint hidden-out matrix and the untied target
unembedding. Folding these weights and the bias on the CPU in FP64 leaves one BF16 vocabulary
projection per proposal. One kernel adds the bias and base scores, applies the FP32 softcap,
and writes the rounded BF16 logit row. The device compares all 200,058 valid vocabulary entries.

The draft Q, O and MLP projections stream DRAM-sharded BF16 weights. K and V use tuned one-dimensional
matmuls. The projections use HiFi4 with FP32 accumulation. Norms, nonlinearities and
attention statistics use FP32. The device context and draft tests warm both graphs before
capturing them.

## Target state

Target K/V and convolution inputs are indexed by absolute position. Global K/V uses one slot per
position. Sliding K/V uses a 1,088-slot ring for a 1,024-key window, so writes for future verify rows
leave earlier queries' keys intact. The reader reconstructs B1's 1,024-slot tile order before
scoring. At the window edge it reads faces from two physical tiles to retain B1's FP32 split and
reduction order.

Each four-tap convolution reads its three earlier inputs from a 32-row FP32 ring and writes
its current input to that ring. Prefill writes its final four inputs into the same slots. Both
decode paths share the per-visit rings.

The verifier groups previous-pass message mixes with independent attention input mixes on
disjoint cores. It batches the Q/K/V message controllers across rows and retains their B1
reduction order. At a short prefix the full four-row verifier replays in 36.45 ms.

One launch packs the Q, K and V inputs into BF16 tile rows. The dynamic taps reuse the
packed K and V inputs where their layouts match. Q, K, V, the dynamic taps and O each
use one sweep of their B1 weights. Head preparation and attention read
the query's row directly from the packed projections. The output gate writes into that same row
of the O projection's input shards. One attention launch per head group runs every verify row
with its B1 split and FP32 reduction order. Short prefixes fit on disjoint workers for each
query. At longer positions the queries reuse a KV head's workers and synchronize before
reusing its circular buffers.

The post-attention and post-MLP convolution and mix epilogues run on disjoint core groups.
Each row reads earlier inputs within the block directly from the packed projection or expert
output. Earlier positions come from the FP32 history ring. Attention preparation batches the
head work and face-row cache writes. Repeat visits batch the MLP message down projections
and blend epilogues within the worker grid.

The router projects all verify rows in one program. Each row has its own top-8 leader and keeps
B1's K-stripe sum and tie-breaking order. One mapping program assigns compact rows in expert-ID order.
The grouped expert path reads one gate/up and down slice per distinct routed expert. It keeps
B1's eight routes and FP32 weights per row, streams the existing BFP8_B weight pack at HiFi4 and
combines the results in the original routing-slot order. Each weight reader uses a two-block
buffer with one read in flight. For two to four rows, the down workers retain the per-expert
BF16 outputs in local buffers and combine them in the same launch. The full 20-expert path
moves 169 MB in 0.443 ms, including the route maps and the combine.

A round accepts a prefix by advancing the position. Later rounds or B1 overwrite rejected
positions before those positions can enter a live window. The continuation ID remains pending
for the next round.

## Round replay

The round allocates scratch and warms each width before capture. The draft transformer and base
head use a trace shared by all widths. Markov scoring, verification and the accepted-context
projection have separate traces on the same command queue. Prefill, B1, the verifier and the
drafter share one width-sharded BF16 head workspace.

The target publishes the next anchor, absolute position and adaptive width. It selects the width
using replay costs measured at startup and a decaying estimate of proposal acceptance. With
estimated conditional acceptance `a`, a width `R` yields an expected `1 + a + ... + a^(R-1)` IDs.

An adaptive request starts with two rows if both its probe overhead and the cost of building
draft context fit 0.3 ms per allowed decode ID. The prefill bucket schedule and startup
measurements of each bucket's context trace give the context cost. Each round charges its estimated replay time and credits the B1 time of its emitted
IDs. A wider round must fit the remaining budget even if every proposal is rejected. The
budget allows one two-row probe. Width one switches the request to B1.

The host queues an asynchronous read of the 64-byte result, followed by the context update and
the next shared draft trace when another speculative round fits and pays off at the lowest possible acceptance. The draft starts from the
device-published anchor and position. The host uses the returned width to submit the matching
scoring and verification traces while the draft runs. Near the short-prefix boundary or the
output limit, the next launch waits for the accepted position.

## Device checks

The component tests run through the pinned dev image.

```sh
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/test_mtp_block.py --compare-rope"
scripts/dev_run.sh "OMP_NUM_THREADS=8 python -u tests/test_mtp_experts.py"
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/test_mtp_router.py"
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/test_mtp_attention.py"
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/test_mtp_decision.py"
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/test_mtp_head.py"
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/test_mtp_verify.py"
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/test_mtp_round.py --adaptive"
INK_LMAX=2048 scripts/bench_decode.sh
```

| Check | Result |
|---|---|
| Three proposals, 1,088-slot draft global prefix | 5.85 ms median, 5.89 ms p95 |
| Draft proposal IDs on two synthetic tap fixtures | 6/6 against CPU |
| Four router rows with staging, shared | 34.68 µs median, 42.16 µs p95 |
| Four expert rows, union of 20 | 0.443 ms median, 0.446 ms p95 |
| Four BF16 target head rows, serial | 7.708 ms median, 7.742 ms p95 |
| Four BF16 target head rows, shared | 2.030 ms median, 2.058 ms p95 |
| Indexed attention with future writes | 208/208 bitwise-equal rows |
| Batched attention against serial B1 | 88/88 bitwise-equal rows |
| Rejected suffixes overwritten by B1 | 18/18 bitwise-equal continuations |
| Four-row target verify at `lmax=1120` | 36.45 ms median, 36.61 ms p95 |
| B1 greedy in the same job | 19.14 ms median, 19.16 ms p95 |

Draft timing measures replay without context extension and host transfers. Expert timings measure
one layer, including the device route maps and the combine.

The serial verifier check compares logits and taps at the local window boundary, the physical
ring wrap and the global cache limit. It also runs B1 with alternate continuations after each
possible rejection and compares with a serial run of the same accepted prefix.

Adaptive rounds reproduce the B1 IDs on 32-token continuations, including a prompt
that reaches the 1,024-key window boundary. Times exclude prefill and draft context priming.

| Prompt | B1 median / p95 ms per ID | MTP median / p95 ms per ID |
|---|---:|---:|
| Chat | 19.28/19.32 | 12.60/12.60 |
| Python | 19.25/19.28 | 13.59/13.60 |
| Korean | 19.16/19.16 | 19.63/19.64 |
| Ring wrap | 19.72/19.73 | 16.16/16.16 |

## HTTP serving

`--mtp` enables round replay for greedy requests without penalties. Each request can select B1
with `"mtp": false`. See [serving.md](serving.md#speculative-decoding) for the checkpoint file and
launch command.

`tools/bench_server.py --compare-mtp --runs 3 --max-tokens 64` compares streamed responses
with both prefill buckets resident and an 8,192-token context. It checks the stream choices
and usage for each run. The HTTP results and the default are in [benchmarks.md](benchmarks.md#adaptive-greedy-mtp).
