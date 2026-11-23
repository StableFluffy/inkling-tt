# DSpark draft and verification kernels

The Inkling-2-Mini checkpoint ships a three-layer DSpark drafter in `mtp.safetensors`. The device
drafter uses target features to propose several tokens for one pending token. A lossless round
requires the target to reproduce B1 logits and state for each proposed input before committing the
matching prefix. The verifier processes each visit in layer order across the block. It
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
committed context and the entire transient block. RoPE uses the block's absolute positions. The
transient self-projected K/V occupy separate slots from the feature-projected prefix.

Rows 1 through K predict the K proposals at their own positions. The block reads the target
unembedding once for their BF16 base scores. A copy from the width-sharded head output collects
only each proposal's live row. The Markov head starts each round with a zero
recurrent state and uses the previous ID, the draft hidden row and the last context mixture. Its
256-wide output projects through the checkpoint hidden-out matrix and the untied target
unembedding. Folding these weights and the bias on the CPU in FP64 leaves one BF16 vocabulary
projection per proposal. The device compares all 200,058 valid scores after the BF16 score add and
softcap.

The draft MLP streams DRAM-sharded BF16 weights. The projections use HiFi4 with FP32 accumulation. Norms, nonlinearities and
attention statistics use FP32. The device context and draft tests warm both graphs before
capturing them.

## Target state

The attention overlay reads committed KV in its ordinary B1 slot order. It substitutes staged
BF16 rows through the current query's position. Future staged rows remain invisible, including at
the point where the local ring wraps. The attention partitions and FP32 reductions are the same
as for a normal B1 step.

The verifier packs the attention inputs into one BF16 tile row. Q, K, V, the dynamic taps and
O each use one sweep of their B1 weights. Head preparation and the causal attention overlay read
the query's row directly from the packed projections. The output gate writes into that same row
of the O projection's input shards.

The router projects all verify rows in one program. Each row has its own top-8 leader and keeps
B1's K-stripe sum and tie-breaking order. The grouped expert path reads one gate/up and down slice per distinct routed expert. It keeps
B1's eight routes and FP32 weights per row, streams the existing BFP8_B weight pack at HiFi4 and
combines the results in the original routing-slot order.

The commit kernel copies the accepted rows' KV into the local rings and global slots. Each
convolution history takes its newest inputs from the accepted prefix and its remaining lags from
the previous history. The continuation ID remains pending for the next round.

## Round replay

The round allocates scratch and warms the device programs before capture. The draft and target
traces replay on the same command queue. The target stages the pending token and draft IDs,
verifies them, and commits the longest matching prefix.
The drafter and target reuse the same width-sharded head input and output buffers.

The round extends the draft cache from accepted target taps, then publishes the next pending ID
and position for the following replay. The host reads the emitted IDs from one 64-byte result.

## Device checks

The component tests run through the pinned dev image.

```sh
scripts/dev_run.sh "OMP_NUM_THREADS=8 python -u tests/test_mtp_block.py"
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/bench_mtp_acceptance.py --steps 64 --proposals 4"
scripts/dev_run.sh "OMP_NUM_THREADS=4 INK_PREFILL_EXPERT_DOUT_L1=480 python -u tests/bench_mtp_acceptance.py --steps 64 --long-context"
scripts/dev_run.sh "OMP_NUM_THREADS=8 python -u tests/test_mtp_experts.py"
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/test_mtp_router.py"
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/test_mtp_attention.py"
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/test_mtp_commit.py"
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/test_mtp_head.py"
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/test_mtp_verify.py"
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/test_mtp_round.py"
INK_LMAX=2048 scripts/bench_decode.sh
```

| Check | Result |
|---|---|
| Three proposals, 1,088-slot draft global prefix | 8.86 ms median, 8.94 ms p95 |
| Draft proposal IDs on two synthetic tap fixtures | 6/6 against CPU |
| Four router rows with staging, shared | 34.68 µs median, 42.16 µs p95 |
| Four expert rows, union of 20 | 0.466 ms median, 0.470 ms p95 |
| Four BF16 target head rows, serial | 8.257 ms median, 8.444 ms p95 |
| Four BF16 target head rows, shared | 2.155 ms median, 2.311 ms p95 |
| Staged local and global attention | 128/128 bitwise-equal rows |
| Traced target state commits | 56/56 equal to sequential updates |
| Four-row target verify at `lmax=1056` | 50.08 ms median, 50.12 ms p95 |
| B1 greedy in the same job | 20.06 ms median, 20.12 ms p95 |

Draft timing measures replay without context extension and host transfers. Expert timings measure
one layer, including the device route maps and the combine.

The serial B1 oracle checks 64 generated IDs on each acceptance fixture. These runs use 8,192
prefix slots for K=3 and 2,048 for K=4.

| Prompt | Input tokens | K=3 emitted per round | K=4 emitted per round |
|---|---:|---:|---:|
| Chat | 33 | 3.200 | 3.368 |
| Python | 35 | 3.368 | 3.765 |
| Korean | 43 | 2.370 | 2.783 |
| Summarization | 7,825 | 2.667 | — |

The 7,825-token fixture uses an 8,192-slot draft global prefix and takes 9.20 ms median and
9.29 ms p95 for three proposals.

The traced verifier matches serial B1 logits and taps bitwise at four starting positions. Committing any prefix length reproduces serial B1 KV and convolution histories,
including a block that spans the local ring wrap and one that ends at the global cache limit.

The shared projections cut four-row verification from 68.11 to 50.08 ms in the same job. The
attention and row epilogues still run once per verify row. The OpenAI server uses the B1 decode path.

The three-proposal round reproduces the B1 IDs on 32-token continuations, including a prompt
that reaches the 1,024-slot ring wrap. Times exclude prefill and draft context priming.

| Prompt | B1 median / p95 ms per ID | MTP median / p95 ms per ID |
|---|---:|---:|
| Chat | 20.14 / 20.20 | 16.25 / 16.29 |
| Python | 20.17 / 20.20 | 17.48 / 17.49 |
| Korean | 20.17 / 20.19 | 24.44 / 24.46 |
| Ring wrap | 20.67 / 20.77 | 22.04 / 22.04 |
