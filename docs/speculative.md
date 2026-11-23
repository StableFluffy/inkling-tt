# DSpark draft and verification kernels

The Inkling-2-Mini checkpoint ships a three-layer DSpark drafter in `mtp.safetensors`. The device
drafter uses target features to propose several tokens for one pending token. A lossless round
requires the target to reproduce B1 logits and state for each proposed input before committing the
matching prefix.

## Draft context

The four BF16 inputs are the incoming partials at chronological visits 5, 23 and 41 and the final
normalized state. `InklingTT2` and `Prefill` can copy these rows into persistent device buffers.
The first three taps precede physical layers 5, 11 and 17 on passes 0, 1 and 2 respectively. Prefill
captures the complete chunk before selecting the last row for the target head. Its descriptor
identifies the rows to append to the drafter's context.

Each draft layer forms its own FP32 mixture of the taps with checkpoint fusion weights, then applies
the layer's weighted RMSNorm. The layer projects BF16 K and V from the mixture and applies a
learned K norm and NeoX RoPE to K. The first and third layers keep 1,024-slot rings. The middle
layer keeps the full prefix. A fourth tap mixture supplies the Markov context from the last
committed input.

## Draft scores

The draft block starts at the first uncommitted position. It embeds the pending target ID in row 0
and fills subsequent rows with the checkpoint mask embedding. Each query attends to its layer's
committed context and the entire transient block. RoPE uses the block's absolute positions. The
transient self-projected K/V occupy separate slots from the feature-projected prefix.

Rows 1 through K predict the K proposals at their own positions. The block reads the target
unembedding once for their BF16 base scores. The Markov head starts each round with a zero
recurrent state and uses the previous ID, the draft hidden row and the last context mixture. Its
256-wide output projects through the checkpoint hidden-out matrix and the untied target
unembedding. Folding these weights and the bias on the CPU in FP64 leaves one BF16 vocabulary
projection per proposal. The device compares all 200,058 valid scores after the BF16 score add and
softcap.

The kernels use BF16 projections at HiFi4 with FP32 accumulation. Norms, nonlinearities and
attention statistics use FP32. The device context and draft tests warm both graphs before
capturing them.

## Target state

The attention overlay reads committed KV in its ordinary B1 slot order. It substitutes staged
BF16 rows through the current query's position. Future staged rows remain invisible, including at
the point where the local ring wraps. The attention partitions and FP32 reductions are the same
as for a normal B1 step.

The grouped expert path reads one gate/up and down slice per distinct routed expert. It keeps
B1's eight routes and FP32 weights per row, streams the existing BFP8_B weight pack at HiFi4 and
combines the results in the original routing-slot order.

The commit kernel copies the accepted rows' KV into the local rings and global slots. Each
convolution history takes its newest inputs from the accepted prefix and its remaining lags from
the previous history. The continuation ID remains pending for the next round.

## Device checks

The component tests run through the pinned dev image.

```sh
scripts/dev_run.sh "OMP_NUM_THREADS=8 python -u tests/test_mtp_block.py"
scripts/dev_run.sh "OMP_NUM_THREADS=8 python -u tests/test_mtp_experts.py"
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/test_mtp_attention.py"
scripts/dev_run.sh "OMP_NUM_THREADS=4 python -u tests/test_mtp_commit.py"
```

| Check | Result |
|---|---|
| Three proposals, 1,088-slot draft global prefix | 9.99 ms median, 10.03 ms p95 |
| Draft proposal IDs on two synthetic tap fixtures | 6/6 against CPU |
| Four expert rows, union of 20 | 0.485 ms median, 0.493 ms p95 |
| Staged local and global attention | 128/128 bitwise-equal rows |
| Traced target state commits | 56/56 equal to sequential updates |

Draft timing measures replay without context extension and host transfers. Expert timings measure
one layer, including the device route maps and the combine.

With the CPU study's estimated 2.307 emitted tokens per three-proposal round, a 20.0 ms B1
baseline allows about 46.1 ms per round. At a 20-expert union, extrapolating the isolated
expert replay to 48 visits gives 23.3 ms. Adding the 12.9 ms B1 non-expert budget and the draft
replay reaches 46.2 ms before additional verification and commit work.

The OpenAI server uses the B1 decode path. Serving qualification requires bitwise agreement
with B1 across the full target and its committed state.
