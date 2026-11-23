# Row epilogues (rowvm)

`inkling_tt/rowvm.py` and `inkling_tt/kernels/rowvm/` implement the small row operations that sit
between the matmuls of a decode visit: norms, attention-residual mixing, head clips, depthwise
convolutions, routing, the MoE activation and combine, the message gates and the readout. Each
public call launches one `ttnn.generic_op`, and the model calls them so that each chain of row work
between two matmuls is one program. The thing to remember: in a trace, a program costs a few
microseconds whatever it computes, so the decode step is fast only if the number of programs is
small.

## Why it exists

Batch-1 decode works on one logical row of 2304 values. Built from stock TTNN ops, a weighted
RMSNorm is three programs, an attention-residual read with four queries is about eighteen, and
router top-8 with its weights is about fifteen. Each of those programs moves a few kilobytes, so its
time is mostly launch and dispatch, not math.

Measured in isolation on the tested card:

| Chain | Stock programs | Stock µs | rowvm µs |
|---|---:|---:|---:|
| Weighted RMSNorm, 2304 | 3 | 32 | 6 |
| Head clip with `tau`, 24x128 | 8 | 32 | 4 |
| Static conv, 2304 channels | 9 | 53 | 8 |
| Attention-residual read, 4 queries, 7 bank rows | 18 | 140 | 16 |
| Router top-8 and weights | 15 | 105 | 8 |

rowvm replaces each chain with one program that keeps the row in L1, does all arithmetic in FP32,
and rounds to BF16 only where the HF model rounds.

## Programs and regions

rowvm is not an interpreter. It is a fixed set of 20 program kinds that share one reader/writer
kernel (`io.cpp`) and one compute kernel (`compute.cpp`). The kind and its options are compile-time
arguments, so tt-metal compiles a separate binary for each combination and the unused branches
disappear.

- **Compute arguments.** Up to six integers: the program kind (`_RMS`, `_MIX`, `_CONV_MIX`, ...),
  the stripe width, a flag word (weighted, BF16 output, residual, bank append, ...), and up to three
  kind-specific values such as a clip threshold in FP32 bits or the number of bank rows.
- **Regions.** Each operand is a `_Region`: a tensor plus a window (start row and column, width, row
  count) and the offsets one task adds to it (`row_step`, `col_step`). Column offsets and widths are
  multiples of 16, so the reader moves whole 16-element face rows of a TILE tensor into dense L1
  scratch. A row-major tensor is read with one transfer per stripe instead of one per face.
- **Tasks.** A program runs on `tasks` logical cores taken in row-major order from the device grid.
  Each core derives its task index from its logical coordinates, then offsets its regions. Wide rows
  split into stripes of 64 to 256 columns, one per task.
- **Collectives.** Programs that reduce across stripes (mixes, wide RMS norms, message blends) send
  FP32 partials to the first stripe of their group through an L1 mailbox and a semaphore. That
  leader computes the softmax or RMS factor and writes the coefficients back to every stripe.
  Semaphores reset after each round, so trace replays reuse the same buffers.

In a convolution mix, the reader issues the branch, history and taps together. It reads the bank
and query while the convolution runs. A weighted mix over seven bank rows takes about 13 µs
with this schedule.

Reader circular buffers are numbered from 0 in operand order and writer buffers from 16. The compute
kernel unpacks BF16 operands into FP32 DEST and unpacks FP32 operands exactly.

Some chains do not fit this shape and have their own kernels: the fused router, the grouped mat-vec,
the message down projection, the head tail, token feedback, the relative bias and the sink tile.
They follow the same conventions (FP32 math, one launch, addresses as runtime arguments).

## Operations

| Python entry | Kernels | Work |
|---|---|---|
| `rms_norm` | `io`, `compute` | RMSNorm of a row, optional weight |
| `attnres_mix` | `io`, `compute` | Depth softmax over the block bank, 1-4 queries, optional weighted RMSNorm |
| `attnres_mix(convolution=...)` | `io`, `compute` | Branch conv, residual, bank append and the next mix |
| `depthwise_conv`, `conv_residual` | `io`, `compute` | Four-tap conv with in-place FP32 history |
| `head_norm`, `head_norm_from_row`, `kv_norm_to_shard` | `io`, `compute` | Per-head L2 clip or RMSNorm, optional `tau` |
| `attention_heads`, `update_kv`, `prepare_attention` | `io`, `compute`, `sink_tile` | Q, R and sink from the packed projection; K/V conv, norm and cache write |
| `gate_from_heads`, `sigmoid_gate`, `add` | `io`, `compute` | Elementwise gate and residual |
| `relative_bias`, `relative_bias_from_position` | `relative_*` | Learned relative bias for the stock attention path |
| `router`, `fused_router` | `io`, `compute`; `router_*` | Sigmoid, biased top-8, unbiased weights; fused version includes the projection |
| `moe_hidden`, `moe_combine` | `io`, `compute` | Softcapped SwiGLU and weighted sum for the stock expert path |
| `message_input`, `message_output`, `message_project` | `io`, `compute` | Message-gate input rows, FP32 up projections, blend and norm |
| `message_down`, `matvec_group` | `msgdown_*`, `matvec_*` | Message-gate down projection with its input norms; grouped BF16 mat-vec |
| `head_tail`, `token_feedback` | `head_tail`, `feedback` | Logits row and masked argmax; next step's inputs on device |

## Chains in one visit

The fused decode graph in `inkling_tt/model2.py` uses these chains for a first-pass sliding-window
visit:

```
attention input row (usually written by the previous visit's last program)
  -> Q/relative/sink/gate matmul, K matmul, V matmul
  -> prepare_attention      Q, R, sink; K/V conv, clip/RMS, cache write
  -> attention decode       (inkling_tt/attn_decode.py)
  -> O matmul
  -> attnres_mix + conv     post-attention conv, residual, bank append, MLP mix, norm
  -> fused_router
  -> expert gate/up, expert down
  -> attnres_mix + conv     post-MLP conv, residual, next visit's attention mix, norm
```

Global layers read three attention inputs with their own three-query mix at the start of the visit
and add two small matmuls for the dynamic conv taps. Repeat passes of the looped core add a
four-query mix over the previous pass's bank, one `message_down` launch for the Q, K and V gates,
`qkv_message_projections` for their blends, and one more down/blend pair for the MLP gate. The
readout is the LM head matmul, `head_tail` and, for greedy decoding, `token_feedback`.

Independent programs on disjoint cores can share one launch through `_Group`: `prepare_attention`
runs the Q, R and sink programs and both K/V cache programs together, and `qkv_message_projections`
runs three gate blends together.

## Launching and tracing

`_program` caches the structural program descriptor, keyed by the compute arguments, the region
layouts and the core list. Each call writes the current buffer addresses as common runtime arguments
and calls `ttnn.generic_op`, so the TTNN program cache also hits when the same layout is used with
different buffers.

A trace records those addresses. Allocate every rowvm output and persistent input before capture,
keep them alive for the life of the trace, and run each distinct chain once eagerly so its binary is
compiled. The convolution and K/V programs update their histories and caches in place;
`InklingTT2.capture` resets them after warmup and capture.

## Constraints

- **One logical row.** Rows are `[1,1,1,N]` with N a multiple of 16 up to 4608; head tensors are
  `[1,1,H,d]` with H in {8, 24} and d in {16, 128}; expert rows are `[1,8,1,W]` or `[1,1,8,W]`. A
  bank holds at most 32 FP32 rows and a mix takes 1-4 queries.
- **Layouts.** Inputs are interleaved TILE tensors with standard 32x32 tiles, or interleaved
  row-major rows whose windows are 64-byte aligned. Sharded tensors are accepted only where the call
  says so (the packed attention projection, K/V shards, message-gate rows that feed a sharded
  matmul). Outputs must not alias inputs; the conv history and bank are the only in-place operands.
- **L1 budget.** A stripe is at most 1024 values wide, so each core holds one dense page per operand
  row (4 KB in FP32); the L1 a program needs grows with the bank and query rows, not with the row
  width. Persistent per-layer constants (norm weights, taps, queries, router bias) are row-major
  rows in DRAM: an interleaved L1 tensor reserves its page in every L1 bank, and 24 layers of them
  do not fit. Only model-wide scratch rows, such as the block bank and the residual rows, live in
  L1.
- **FP32 math.** Compute runs at HiFi4 with FP32 DEST and uses the accurate SFPU exp, tanh, sigmoid,
  sqrt and rsqrt. Reciprocals take two Newton steps. BF16 outputs round to nearest-even with a
  software helper, because the Blackhole conversion instruction rounds ties away from zero. Results
  can differ from a PyTorch FP32 reference by a few ulps because the stripe reduction order differs.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `INK_MIX_CHUNK` | 128 | Stripe width of `attnres_mix` |
| `INK_ROUTER_STRIPES` | 24 | K stripes of `fused_router` |
| `INK_MATVEC_STRIPES` | 18 | K stripes per projection in `matvec_group` |
| `INK_ROWVM_RECT` | 1 | `0` passes the task index as a per-core runtime argument instead of deriving it from coordinates |
| `INK_ROWVM_ZONES` | unset | `1` compiles device-profiler zones into the kernels |

## Adding an operation

1. Append the program name to the kind tuple at the top of `rowvm.py` and to the `Program` enum in
   `kernels/rowvm/compute.cpp`, at the same position.
2. Write the arithmetic in `compute.cpp` as an SFPU routine on FP32 DEST vectors, reusing the
   helpers there (`sum32`, `reciprocal`, `rounded`, the striped RMS steps), and dispatch to it from
   `kernel_main`.
3. Add the Python entry: check shapes with `_validate` or `_width`, allocate the output with
   `_output`, describe each operand as a `_Region` (or `_stripe` for striped rows), and call
   `_run(compute_args, readers, writers, tasks)`.
4. If the operation reduces across stripes, add it to `_COLLECTIVES` and use the existing exchange
   rather than a new one.
5. Test against PyTorch in FP32 and against BF16 rounding of the device's own FP32 result, then
   replay it in a trace with new inputs written to the same addresses. `tests/test_rowvm.py` follows
   this pattern; `scripts/run_tests.sh kernels` runs the rowvm tests.
6. Call the new entry in the eager graph before `capture()` so it compiles during warmup.

Size core counts from `device.compute_with_storage_grid_size()` as the existing entries do. The
supported grids are 11x10 to 13x10 ([hardware.md](hardware.md)).
