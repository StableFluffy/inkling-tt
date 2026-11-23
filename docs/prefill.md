# Prefill

Prefill runs the prompt through the decode model's weights and caches in fixed-shape chunks, each replayed from a trace. Two buckets, 128 and 2048 rows, stay resident; every chunk advances all 48 layer visits and writes the K/V caches and convolution histories that the decode trace reads next. The rule to remember: every prefill program and buffer is created and run once before the decode trace is captured.

The code is `inkling_tt/prefill.py`, the expert path in `inkling_tt/experts_stream.py`, and the kernels in `inkling_tt/kernels/prefill/`. The model it computes is described in [model.md](model.md).

## Buckets and chunks

`Prefill(m, chunk)` owns one fixed chunk shape: a multiple of 32 between 128 and 2048 rows. Rows past the prompt are padding. A small descriptor in device memory holds the chunk's start position and valid row count, and every kernel reads it, so one trace serves first chunks, middle chunks and short tails. A chunk must start at a multiple of 32.

The server runtime (`serve/runtime.py`) builds two buckets, 128 rows and `--prefill-chunk` (default 2048). `prefill_buckets(buckets, tokens)` picks the bucket for each chunk from the tokens still to run, minimizing `ceil(rest / rows) * (rows + 352)`. The estimate adds a fixed cost equivalent to 352 rows to the cost of processing the full bucket. In practice 512 or fewer remaining tokens run as 128-row chunks and anything longer takes a 2048-row chunk.

| Prompt tokens | Chunks |
|---:|---|
| 100 | 1 x 128 |
| 300 | 3 x 128 |
| 1,024 | 1 x 2048 |
| 2,100 | 1 x 2048 + 1 x 128 |
| 8,192 | 4 x 2048 |

The split does not change the result. `tests/test_prefill_buckets.py` checks that a 2,100-token prompt in mixed chunks and a 4,100-token prompt in 2048-row chunks give logits bit-identical to running the same prompts in 128-row chunks.

## One chunk

The host normalizes the chunk's embedding rows to BF16, computes the global-attention scale `tau` for each row, and writes both with the descriptor into persistent device buffers. The body trace then runs the 48 visits in loop order, and a final program copies the hidden state of the last valid row into a one-row buffer. `prefill_buckets` runs the readout trace, the LM head on that row, only after the last chunk; the host reads the logits and applies the softcap.

Each visit runs these steps over all rows of the chunk:

1. Depth-mix the bank (and, on repeat passes, blend in the messages), then apply the learned input norm.
2. Project Q, R, sink and gate in one bundle; clip the heads and apply `tau` on global layers.
3. Project K and V, run the stencil convolution, clip K and RMS-normalize V per head.
4. Run tiled attention against the cache plus the chunk's own keys, then commit the chunk's K and V.
5. Apply the sigmoid gate, the O projection and the post-attention convolution; add the residual.
6. Depth-mix for the MLP read, normalize, route on device, run the experts, apply the post-MLP convolution and add the residual.

## Dense projections

Prefill reads the decode weights in place; there is no second copy. Most schedules keep the decode programs' accumulation order, so prefill and decode project a row to the same bits.

- Q bundle (2304 x 6560) and O projection: each worker reads its decode weight columns once per chunk and the activation is broadcast along rows of cores (`dense.linear_prefill_resident`).
- Message-gate down projection (4608 x 64): eight K partitions emit FP32 partials, reduced in partition order (`dense.message_prefill_down`). This one is close to decode but not bit-identical.
- K and V: one full-chunk `ttnn.linear`.
- Other DRAM-sharded weights: the decode program runs once per 32-row block.
- Router: BF16 operands in 96-wide K blocks, with the partials summed in FP32. For some tokens the score margin between the eighth and ninth expert is smaller than the rounding drift of one full-K accumulation.

## Routing and experts

Routing stays on the device. `route_prefill` adds the correction bias to the sigmoid scores, then `route.cpp` picks eight experts per token (exact ties go to the lower expert ID) and normalizes the unbiased scores in FP32. A second phase on one core builds the maps for the expert kernels: where each expert's rows end, which token fills each compact row, and where each token's eight results land. Every expert's row count is rounded up to whole blocks of 8 rows (128 bucket) or 32 rows (512 rows and up), so buffer shapes depend only on the chunk size and new routes never compile a new program.

`moe_prefill` runs four stages: gather the routed rows into expert-ordered compact tiles, gate/up with the softcapped SwiGLU, down, and combine. The projections stream the same bank-local BFP8_B expert pack that decode uses; one copy of the 26 GB pack fits on the card, a second layout would not. The 128-row bucket uses compact 8-row tiles and keeps one 13 KB weight read in flight per worker.

At 512 rows and above the projections run as lanes of 18 workers, seven lanes on a 13x10 grid and six on 11x10. Each lane takes a contiguous range of compact blocks and streams the weights of the experts in that range. Lane members take turns reading an activation block and multicasting it to the lane. A bounded read window per worker limits contention at the DRAM banks.

The combine sums each token's eight expert outputs, weighted by `gamma`, in FP32 and in expert-ID order, and writes BF16. Math is HiFi4 with FP32 accumulation and the BF16 boundaries of [model.md](model.md#cast-points).

## Attention

Prefill attention is a custom kernel (`attention_io.cpp`, `attention_compute.cpp`). Stock SDPA prefill accepts one constant sink per head, while Inkling's sink varies per query, and it would need the relative bias materialized as a full mask.

One task covers one query head and one 32-row query tile; tasks alternate early and late tiles so that contiguous task ranges have similar causal spans. The reader walks key tiles in causal order. Keys from earlier chunks come from the visit's global cache or its 1024-row ring; keys of the current chunk come from the freshly projected K and V. For the relative bias, the compute kernel multiplies the padded relative query by the layer's reversed `[32, 1024]` distance table, giving FP32 logits for distances 0-1023. The reader and writer then shear those into each score tile, even and odd lags split between them. Global keys farther than 1023 positions get no bias.

The softmax takes two sweeps over the keys. The first tracks the running maximum and sum in FP32, starting from the row's sink logit. The second forms probabilities, rounds them to BF16 (round to nearest even), and accumulates their product with V in FP32. Scores from the first sweep stay in L1 where they fit: a 33-tile buffer covering the window on sliding visits, and the most recent 176 tiles on global visits at 2048 rows on the tested grid. Smaller worker grids retain 148 tiles at that chunk size, and chunks up to 1024 rows retain up to 97. With the standard tt-metal dispatch ring, the 2048-row bucket retains 188 tiles on the tested grid and 160 on smaller grids. The second sweep recomputes scores beyond the retained range. The output is written token-major, ready for the gate and O projection.

With both buckets resident and the standard ring, retaining 188 score tiles instead of 160 cuts 32k-token prefill time by about 0.15 s and uses an extra 112 KiB of L1 per attention worker. The wide bucket's global attention reader also
issues pairs of historical key tiles between read barriers. Each pair stays within the circular
buffers, one saved-score range and one group of eight keys for the value product. This saves
another 0.28 s on a 32k-token prompt.

`state.cpp` commits the chunk's K and V after all attention reads of the visit. Global visits write absolute positions; sliding visits write the ring at `position mod 1024`. A 2048-row chunk covers the ring twice, so only the newest tile writes each slot, and a partial newest tile takes the rows past its end from the tile one ring length earlier in the same chunk.

## Convolution, depth mixing, message gates

- The stencil kernel (`conv_io.cpp`, `conv_compute.cpp`) applies the four-tap causal convolution, static or dynamic, to a whole chunk. It reads the visit's three-row FP32 history, accumulates in FP32, and updates the history in place in the layout decode reads. At position 0 it treats the history as zero.
- The heads kernel (`heads_io.cpp`, `heads_compute.cpp`) clips or RMS-scales each 128-channel head in FP32 and writes head-major tiles; global queries are rounded to BF16 and multiplied by their row's `tau`.
- Depth mixing computes RMS and the depth logits with TTNN ops and the weighted sum over bank entries with `mix_compute.cpp`, for one, three or four query rows.
- The message-gate kernel (`gate_compute.cpp`) evaluates `u * (1 - tanh(h W_f)) + m * tanh(h W_g)` in FP32 for one output tile.

## Handoff to decode

Prefill and decode share the visit state of `InklingTT2`: the 48 K/V caches, the 192 convolution histories and the position. After the last chunk the cache holds the whole prompt, the readout gives the first output token, and decode continues at `position = len(prompt)`. Cache slots past the position are masked, so a new request only zeroes the convolution histories (`InklingRuntime.reset`).

## Capture order

A trace replays recorded commands at fixed device addresses. A buffer allocated while a trace exists can land on memory that the trace uses for its intermediates, so every buffer must exist before `m.capture()`. Every program must also have run once, so that its binary is compiled and cached before capture:

```python
m = InklingTT2(device, weights, default_precision(), lmax=32768)   # serve/runtime.py precision; allocates KV
buckets = [Prefill(m, 128), Prefill(m, 2048)]                      # allocate, compile, run once, reset
m.capture()                                                        # decode trace, then each bucket's traces
logits, pos = prefill_buckets(buckets, ids)
```

Each `Prefill` constructor refuses to run after capture. It allocates its buffers and expert workspace, runs the body and readout eagerly on a dummy chunk, resets the model state, and chains itself onto `m.capture`. The chained call warms and captures decode first, then a body trace and a readout trace for each bucket in creation order. Close buckets in reverse order with `Prefill.close()` before closing the device.

## Sequence length

`lmax`, a multiple of 32, sizes the eight global caches when the model is built; the 40 sliding visits keep fixed 1024-slot rings. Prompt plus output must fit in `lmax`. The server default and the tested maximum is 32768. Fused attention decode supports global caches up to 32768 slots (`inkling_tt/attn_decode.py`); above that, decode falls back to stock SDPA, which this release does not test. At 32768, the decode trace, both buckets and the on-device embedding table use about 31 GB of the card's 33.8 GB of allocatable DRAM ([hardware.md](hardware.md#memory)).

## Throughput

Chunk time is close to a fixed part plus a part linear in rows. The fixed part includes the expert weights: even a 128-row chunk routes to most of the 128 experts at every visit, so every chunk streams about twice the 26 GB expert pack (48 visits over 24 physical layers). Short prompts are therefore overhead-bound; a 128-token prompt runs at roughly a third of the peak rate. Throughput peaks around one full 2048-row chunk. Beyond that it falls slowly, because the eight global visits attend over a growing prefix; at 32k tokens the rate is roughly 70 % of the peak. Release TTFT figures are in [benchmarks.md](benchmarks.md).
