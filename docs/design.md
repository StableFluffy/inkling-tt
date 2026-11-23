# Decode design

A decode step on the tested grid replays 607 device programs for 48 layer visits and the LM head. Weight streaming accounts for most of the kernel time. A contiguous, bank-local read benchmark reaches 490 GB/s.

Two rules shaped the design:

- Read every weight once per token, at close to peak DRAM bandwidth.
- Launch as few programs as possible. In a trace, each program costs a few microseconds to dispatch and drain, whatever it computes. A direct port of the model issues about 2,400 programs per token, which costs more than the weight reads.

## Where the time goes

Device profile on the tested card (13x10 workers, `lmax` 2048, BFP8_B experts). The table gives medians over three one-token replays. Each row sums the kernels for that work.

| Work | Programs | Kernel ms |
|---|---:|---:|
| Routed experts (gate/up, down) | 96 | 6.90 |
| Q bundle and O projections | 96 | 4.60 |
| LM head | 1 | 1.84 |
| Paired K/V projection and head preparation | 48 | 1.44 |
| Dynamic-tap projections | 16 | 0.09 |
| Attention-residual mixes | 120 | 1.28 |
| Message-gate projections | 96 | 1.21 |
| Router | 48 | 0.46 |
| Fused attention | 48 | 0.42 |
| Readout, layout and small epilogues | 38 | 0.13 |
| **Total** | **607** | **18.38** |

The expert and projection programs account for 14.9 ms, including the head preparation fused into the K/V projection. Firmware spans 18.60 ms per token, with 0.22 ms outside kernels. The larger dispatch ring lets Metal stage the next program while the current one runs. Unprofiled decode timings are in [benchmarks.md](benchmarks.md).

## From the first port to this release

The optimizations use the precision policy in [precision.md](precision.md).

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
| Fused router, head tail, message gates and token feedback (750 programs) | 20.0 |
| Paired K/V, tuned dense projections and batched row reads | 19.3 |
| Larger dispatch ring and fixed read virtual channels | 18.5 |
| Paired K/V and head preparation in one launch | 18.4 |

## The pieces

**Trace with persistent inputs.** `InklingTT2` (`inkling_tt/model2.py`) allocates every intermediate once and captures the whole step with `ttnn.begin_trace_capture`. Per-step data (token embedding row, positions, cache slots, masks) lives in device tensors that the host overwrites before each replay. Anything allocated or compiled after capture starts can alias trace memory, so the model and both prefill buckets run once eagerly before `capture()` is called.

**Dispatch memory.** The standalone runtime reserves about 128 KiB per worker for Metal's kernel-config ring. The larger ring lets dispatch stage more programs while earlier programs are still running. The 2048-row bucket keeps fewer global-attention score tiles and spills more of the expert down projection to DRAM to fit the smaller L1 partition.

**NoC reads.** Decode data-movement kernels pick a read virtual channel from their logical worker row modulo four. This removes dependence on the inherited read command state. Expert weight streams still select their own channels for the bank-local packets.

**Dense projections.** Batch-1 projections are matrix-vector products, bound by weight bandwidth. `inkling_tt/dense.py` picks a DRAM-sharded matmul configuration per shape, so each DRAM bank feeds the workers next to it. The four attention projections of a visit (Q, relative-bias query, sink, output gate) share one fused weight and one program. The paired K/V projection streams both matrices on 64 bank-local workers. On the full grid, 65 other workers prepare attention heads in the same launch. See [dense.md](dense.md).

**Routed experts.** Each visit routes to 8 of 128 experts. The stock indexed `sparse_matmul` reads only the routed experts too, but from tiles interleaved across all DRAM banks and in six programs. `inkling_tt/experts_stream.py` stores each expert as contiguous bank-local BFP8_B runs and reads the eight selected experts in two programs, about 146 µs per visit instead of 234 µs. Routing stays on device: the router writes expert IDs and weights to a buffer the expert kernels read. See [experts.md](experts.md).

**Row epilogues.** Between the matmuls sit chains of small row operations: RMS norms, attention-residual mixing over block banks, head norms and clips, depthwise convolutions, top-8 routing, the MoE activation and combine. As separate TTNN ops they cost 3-5 µs each and dominated the step. `inkling_tt/rowvm.py` compiles each chain into one `generic_op` program that keeps the row in L1 and does its math in FP32. See [rowvm.md](rowvm.md).

**Visit fusion.** Consecutive visits share boundaries: the epilogue that ends one visit also starts the next one, and the convolution history, residual and block bank are updated in the same program. A first-pass sliding visit runs the Q bundle, K/V and head preparation, fused attention, O projection, post-attention mix, router, two expert projections and post-MLP mix.

**Fused attention.** One program per visit computes the learned relative bias, the per-head sink, the masked softmax in FP32, the V product and the sigmoid output gate, and writes its output straight into the input shards of the O projection. It reads only the cache positions that exist, so a large `lmax` does not slow short sequences. See [attention.md](attention.md).

**Readout.** The LM head writes width-sharded logits. One program (`rowvm.head_tail`) copies them into a row-major logits row, masks the padded vocabulary and finds the argmax. For greedy decoding a second one-core program (`rowvm.token_feedback`) copies the chosen token's embedding row from an on-device table into the next step's input and advances the position, so the host does not touch the critical path. `m.greedy(tok, pos, n)` enqueues `n` replays and reads all IDs back at the end.

**Sampling.** The sampled server path applies its filters to the entire head row on device and reads one token ID. A bounded heap on each worker finds local candidates for small exact top-k requests. The sampler uses a radix selection to find score and probability-mass cutoffs for larger sets. Penalty counts stay in a reusable device buffer.

**Row-major state.** The decode bank, residual scratch, embedding row and rowvm constants are stored row-major. A TILE layout stores one row as 16-value faces, so reading a row cost one NoC transaction per 16 values; row-major reads it in one transfer per stripe.

## What did not help

- Merging the expert down projection and the following mix into one program. Outputs were identical, but the merged program lost its rectangular core grid and replay got 0.24 ms slower.
- Keeping rowvm state in L1 instead of DRAM. It saved under 1 µs per mix, and row-major rows saved more.
- Lower precision for speed: BFP4_B experts and HiFi2 math. Both fail the accuracy bar; see [precision.md](precision.md).

## Remaining budget

About 3.5 ms of kernel time sits outside the expert and projection programs. Attention-residual mixes and message gates account for most of that time. The router and fused attention each use under half a millisecond at short context.
