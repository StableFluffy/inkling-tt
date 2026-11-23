# Decode attention

Decode attention runs as one program per visit (`inkling_tt/attn_decode.py`,
`inkling_tt/kernels/attn/`) that computes the learned relative bias, the per-head sink, the masked
FP32 softmax, the V product and the sigmoid output gate, and writes the result straight into the
input shards of the O projection. It reads only the cache slots that hold tokens, so its cost grows
with the position, not with the allocated context `lmax`.

## What Inkling attention needs

Inkling-2-Mini has 24 query heads and 8 KV heads of width 128; three query heads share each KV head.
Layers 5, 11, 17 and 23 attend globally. The other layers use a sliding window of 1024 tokens, the
current one included. With the looped core this gives 8 global and 40 sliding-window visits per
token, each with its own KV cache. There is no RoPE. Instead:

- **Relative bias.** Each query head projects a rank-16 vector R from the attention input. The bias
  for a key at distance `d` is `R · table[d]` for `d < 1024`, where `table` is a learned `[1024,
  16]` matrix per layer. On global layers keys at distance 1024 or more stay visible with zero bias.
- **Sink.** Each head projects one extra logit from the attention input. It joins the softmax with a
  zero value, so it only lowers the weight of the real keys.
- **Clips and norms.** Q and K heads are L2-clipped to norm `2 * sqrt(128)`, R heads to 8; Q and R
  carry learned per-channel scales. V heads get an unweighted RMSNorm.
- **Position scale.** On global layers Q and R are multiplied by `tau(p) = 1 + 0.1 * ln(max((p + 1)
  / 8192, 1))` after clipping.
- **Output gate.** The 3072-wide attention output is multiplied by `sigmoid` of a fourth projection
  of the attention input before the O projection.
- **Short convolutions.** K and V pass through a causal four-tap depthwise convolution before the
  clip and norm. Sliding layers use static taps; global layers add taps projected per KV head from
  the attention input.

The equations and rounding points are in [model.md](model.md).

## Before the kernel

One matmul computes Q, R, the sink logits and the gate logits from the attention input
([dense.md](dense.md)). The model folds the learned Q and R scales into those weights at load time.
Two small matmuls compute the dynamic taps on global visits. On a 13x10 grid,
`rowvm.prepare_attention` runs the paired K/V projection and head preparation in one launch.
The projection workers keep their bank-local placement. A cache worker reads the projected head
after its four column tiles signal completion. On smaller grids, the paired projection runs in
its own launch. Set `INK_FUSED_KV_PREP=0` to use the separate program on the full grid.

The preparation kernels perform three operations ([rowvm.md](rowvm.md)).

- Q and R: clip per head, round to BF16, apply `tau` on global visits and round again, as HF does. R
  is zero-extended from 16 to 32 columns.
- Sink: the 24 logits are copied into one tile.
- K and V: convolve with the FP32 history (updated in place), clip K or normalize V per head, and
  write the BF16 row into the cache at the current position.

The caches are BF16 `[1, 8, L, 128]` tensors in DRAM. Fused single-card sliding attention writes
at `position mod 1088`, leaving 64 slots beyond the 1,024-key window for speculative writes. The
reader maps physical tiles to the original 1,024-slot order for scoring. The tile at the window
edge can contain rows from two physical tiles, one at each end of the live window. Global caches
have `L = lmax` slots. The fused kernel takes multiples of 32 up to 32768, and the model uses stock
SDPA beyond that.

## The fused kernel

`attn_decode.decode` gives each KV head 16 workers. The three query heads of that KV head share
them, and the workers split the live key tiles into contiguous ranges.

**Position-bounded reads.** Every worker reads the position tensor on device and computes `valid =
min(position + 1, window or global capacity)` and the number of live 32-key tiles. Only those tiles are read and only the
splits that have tiles do score work, so a short sequence in a 32k-slot global cache reads a few
tiles. K and V reads rotate their starting DRAM bank across workers so the requests spread over the
eight banks.

**Bias lookup.** For each key tile the relative distances form a 32x32 block that depends only on
`position mod 32` and the tile's distance in tiles. `prepare_table` precomputes all 32 phases of the
table as BF16 tiles, so each bias needs one aligned tile read and one small matmul against R.
Sliding layers wrap distances modulo 1024. On global layers a tile whose keys are all 1024 or more
positions back skips the table read and the bias matmul. The packed table is about 2 MB per layer in
DRAM, built on the first eager call and shared by the layer's visits.

**Scores.** QK and the bias matmul accumulate in FP32 DEST. The kernel adds `-1e30` to keys at or
beyond `valid`, rounds the bias to BF16 nearest-even, and computes `(QK + bias) / sqrt(128)`.

**Softmax.** Each split computes its FP32 maximum and sum with the accurate FP32 exponential. When a
split has more than four key tiles, it streams them in chunks of four and carries a running maximum
and sum. The first split of each head gathers all splits' statistics, adds the sink as one more
logit with zero value, and sends a normalization factor back to each split.

**Values and gate.** Each split rounds its normalized probabilities to BF16 and multiplies them by
BF16 V with FP32 accumulation. Four workers per KV head, one per 32-column slice of the 128-wide
head, reduce the partial outputs, round to BF16, multiply by `sigmoid` of the gate logits and round
again. They write the result into the width-sharded input of the O projection, so no layout
conversion follows.

The two in-program reductions use L1 mailboxes and semaphores between the workers of one head. Eight
heads need 128 workers. On the 13x10 grid all eight run in one launch; on 11x10 and 12x10 the kernel
launches two groups of four heads, with the same splits and reduction order.

For two to four verify rows, a short prefix leaves enough workers in each KV head to give every
query a disjoint group. Each group runs the same split indices and FP32 reductions as B1. As
the key prefix grows, the launch reuses a KV head's workers across queries and synchronizes the
head before reusing its circular buffers.

## Cost

Measured in isolation, without the Q/K/V preparation and the O projection:

| Visit | Position | Stock µs | Fused µs |
|---|---:|---:|---:|
| Sliding window, 1024 slots | 31 | 45 | 9 |
| Sliding window, 1024 slots | 1023 | 45 | 20 |
| Global, 2048 slots | 31 | 64 | 9 |
| Global, 2048 slots | 2047 | 64 | 31 |

The stock column is relative bias, SDPA decode and output gate. The stock cost is flat because it
reads the whole allocated cache; the fused cost follows the position.

## Numerics

The fused kernel keeps scores and softmax statistics in FP32 and rounds the normalized probabilities
to BF16 before the V product, which is where HF eager rounds them. Stock SDPA decode at the pinned
tt-metal commit keeps its score intermediates in BF16 and normalizes after the V product, so the two
paths differ in BF16 bits. Release accuracy figures, measured with the fused kernel, are in
[benchmarks.md](benchmarks.md).

## Stock fallback

`INK_ATTN_DECODE=0` replaces the fused kernel with three programs:
`rowvm.relative_bias_from_position` writes a BF16 bias row over the full cache length with the
causal mask included, `ttnn.transformer.scaled_dot_product_attention_decode` runs with that bias as
`attn_mask`, the sink tile as `attention_sink` and `exp_approx_mode=False`, and
`rowvm.gate_from_heads` applies the gate into the O-projection shards. Stock SDPA decode computes
`softmax((QK + mask) * scale)`, so the model stores the relative table and the sink weights
pre-multiplied by `sqrt(128)`; the fused kernel uses the same tables and applies the scale at the
same point.

The fused kernel is used only on Blackhole with the visit-fused graph, which needs the streaming
experts. `INKLING_EXPERTS=stock` therefore also selects the stock attention path.

## Tests

`tests/bench_attn_decode.py` replays both paths in a trace at positions that cover short prefixes,
tile boundaries, the ring wrap and the global zero-bias region, checks the fused output against a
PyTorch reference, and times both. It runs in `scripts/run_tests.sh kernels`.
