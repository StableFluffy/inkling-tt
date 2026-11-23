# Decode design

A decode step is one trace replay: 750 device programs that run the 48 layer visits of one token and the LM head. The weight-streaming programs take about 15.3 ms of the 19.4 ms of kernel time per token. That is close to what reading the weights costs at the bandwidth a p150a sustains, so the remaining work is in the other 4 ms and in the gaps between programs.

Two rules shaped the design:

- Read every weight once per token, at close to peak DRAM bandwidth.
- Launch as few programs as possible. In a trace, each program costs a few microseconds to dispatch and drain, whatever it computes. A direct port of the model issues about 2,400 programs per token, which costs more than the weight reads.

## Where the time goes

Device profile of one token on the tested card (13x10 workers, `lmax` 8192, BFP8_B experts). Kernel times are sums over all programs of a kind.

| Work | Programs | Kernel ms |
|---|---:|---:|
| Routed experts (gate/up, down) | 96 | 7.1 |
| Q/relative/sink/gate and O projections | 96 | 4.6 |
| LM head | 1 | 2.1 |
| K/V and dynamic-tap projections | 112 | 1.5 |
| Attention-residual mixes | 120 | 1.4 |
| Message-gate projections | 144 | 1.3 |
| Router | 48 | 0.5 |
| Fused attention | 48 | 0.5 |
| Attention preparation (norms, conv, cache write) | 48 | 0.3 |
| Readout, layout and small epilogues | 37 | 0.1 |
| **Total** | **750** | **19.4** |

The first four rows stream weights; together they are the 15.3 ms. The profiled span is about 20.4 ms, so the gaps between programs add about 1 ms. Unprofiled greedy decode runs at about 20.0 ms per token ([benchmarks.md](benchmarks.md)).

## From the first port to this release

Each step kept the precision policy of [precision.md](precision.md); the accuracy checks pass at every step.

| Step | ms/token |
|---|---:|
| Eager TTNN port, one op at a time from Python | 850 |
| Whole step captured as one trace | 129 |
| Relative bias and sinks folded into SDPA decode | 85 |
| DRAM-sharded dense projections | 73 |
| Row-epilogue programs between matmuls (rowvm) | 42.8 |
| Streaming only the eight routed experts | 37.8 |
| Visit fusion (2,418 to 942 programs) | 30.3 |
| Row-major logits and on-device argmax | 24.5 |
| Fused attention decode | 22.3 |
| Fused router, head tail, message gates, device token feedback | 20.0 |

## The pieces

**Trace with persistent inputs.** `InklingTT2` (`inkling_tt/model2.py`) allocates every intermediate once and captures the whole step with `ttnn.begin_trace_capture`. Per-step data (token embedding row, positions, cache slots, masks) lives in device tensors that the host overwrites before each replay. Anything allocated or compiled after capture starts can alias trace memory, so the model and both prefill buckets run once eagerly before `capture()` is called.

**Dense projections.** Batch-1 projections are matrix-vector products, bound by weight bandwidth. `inkling_tt/dense.py` picks a DRAM-sharded matmul configuration per shape, so each DRAM bank feeds the workers next to it. The four attention projections of a visit (Q, relative-bias query, sink, output gate) share one fused weight and one program. See [dense.md](dense.md).

**Routed experts.** Each visit routes to 8 of 128 experts. The stock indexed `sparse_matmul` reads only the routed experts too, but from tiles interleaved across all DRAM banks and in six programs. `inkling_tt/experts_stream.py` stores each expert as contiguous bank-local BFP8_B runs and reads the eight selected experts in two programs, about 155 µs per visit instead of 234 µs. Routing stays on device: the router writes expert IDs and weights to a buffer the expert kernels read. See [experts.md](experts.md).

**Row epilogues.** Between the matmuls sit chains of small row operations: RMS norms, attention-residual mixing over block banks, head norms and clips, depthwise convolutions, top-8 routing, the MoE activation and combine. As separate TTNN ops they cost 3-5 µs each and dominated the step. `inkling_tt/rowvm.py` compiles each chain into one `generic_op` program that keeps the row in L1 and does its math in FP32. See [rowvm.md](rowvm.md).

**Visit fusion.** Consecutive visits share boundaries: the epilogue that ends one visit also starts the next one, and the convolution history, residual and block bank are updated in the same program. This took the step from about 50 to about 20 programs per visit; the fused router, message gates and head tail brought it to about 16.

**Fused attention.** One program per visit computes the learned relative bias, the per-head sink, the masked softmax in FP32, the V product and the sigmoid output gate, and writes its output straight into the input shards of the O projection. It reads only the cache positions that exist, so a large `lmax` does not slow short sequences. See [attention.md](attention.md).

**Readout.** The LM head writes width-sharded logits. One program (`rowvm.head_tail`) copies them into a row-major logits row, masks the padded vocabulary and finds the argmax. For greedy decoding a second one-core program (`rowvm.token_feedback`) copies the chosen token's embedding row from an on-device table into the next step's input and advances the position, so the host does not touch the critical path. `m.greedy(tok, pos, n)` enqueues `n` replays and reads all IDs back at the end.

**Row-major state.** The decode bank, residual scratch, embedding row and rowvm constants are stored row-major. A TILE layout stores one row as 16-value faces, so reading a row cost one NoC transaction per 16 values; row-major reads it in one transfer per stripe.

## What did not help

- Merging the expert down projection and the following mix into one program. Outputs were identical, but the merged program lost its rectangular core grid and replay got 0.24 ms slower.
- A custom bank-streamed K/V projection. It matched the stock kernels' arithmetic rate but was not faster than the tuned DRAM-sharded and 1D matmuls at this size.
- Keeping rowvm state in L1 instead of DRAM. It saved under 1 µs per mix, and row-major rows saved more.
- Lower precision for speed: BFP4_B experts and HiFi2 math. Both fail the accuracy bar; see [precision.md](precision.md).

## Remaining budget

About 4 ms of kernel time and 1 ms of gaps sit outside weight streaming. The largest items are the 120 attention-residual mixes (1.4 ms), attention preparation and fused attention (0.8 ms), the message-gate up projections (0.9 ms, partly FP32 weight reads) and the router (0.5 ms). Shaving single programs has small returns now. The next real gain needs overlap across programs, for example prefetching the next visit's weights on a sub-device while the epilogues run, or a persistent decode kernel.
