# Routed experts

Each decode visit routes the token to 8 of 128 experts. `inkling_tt/experts_stream.py` and
`inkling_tt/kernels/experts/` run those eight experts as two programs per visit that read only the
routed experts' BFP8_B weights, from a pack laid out so that every reader streams long contiguous
runs from the DRAM bank next to it. The routed experts are the largest weight read in a token, so
this path decides most of the decode time.

## The pack

`experts_stream.prepare` takes the converted BF16 tensors of one physical layer, gate/up `[128,
2304, 2304]` (gate rows first) and down `[128, 2304, 1152]`, and uploads them as two
DRAM-interleaved BFP8_B tensors. An interleaved tensor places tile `p` in bank `p % 8`, so the
packer orders tiles as `8 * bank_local_tile + bank`: every reader's chunk is contiguous inside one
bank. The packer moves whole 32x32 tiles, which keeps the face-row exponent groups of TT's BFP8_B
conversion unchanged.

```
gate/up, per expert, per bank (648 tiles)
  group A: 64 channels  = 2 gate + 2 up columns x 72 K tiles   (288)
  group B: 64 channels  = 2 gate + 2 up columns x 72 K tiles   (288)
  half of a shared 32-channel group: 1 gate + 1 up column,
     alternate 6-tile K blocks, the other half in a paired bank  (72)
  each group stored as 6-tile K blocks with all its columns adjacent

down, per expert, per bank (324 tiles)
  output columns bank + 8*j, j = 0..8   x 36 K tiles each
```

Eight banks times two 64-channel groups cover the first 1024 of the 1152 hidden channels; the last
128 are four 32-channel groups split across bank pairs. A layer's pack is 1,082,916,864 bytes, the
same as the stock BFP8_B layout, and 26.0 GB for all 24 layers. The eight routed experts of one
visit are 67,682,304 bytes: 7,776 tiles of 1,088 bytes each.

`prepare` also assigns workers. It asks tt-metal for the worker nearest each DRAM bank
(`get_optimal_dram_bank_to_logical_worker_assignment`) and takes the free workers closest to it,
preferring the forward NoC0 direction. The streamer needs ten workers per bank, which fits every
supported grid.

## Two programs per visit

`moe_decode(x, ids, gamma, handle)` reads the router's outputs from device memory: eight UINT16
expert IDs and eight FP32 weights. No host round trip happens inside a token.

**Gate/up (80 workers).** Each bank has eight local workers: two channel groups times four lanes.
Lane `l` handles routing slots `l` and `l + 4`, so the four lanes cover all eight experts. Sixteen
more workers handle the shared 32-channel groups and alternate K blocks between the two banks of
their pair. For each slot, a worker multiplies the activation by its gate and up columns into FP32
DEST, then applies the activation in the same program: it rounds the gate and up projections to BF16
as the HF model does, computes `4 tanh(g/4) * sigmoid(g) * 25 tanh(u/25)` with the accurate FP32
tanh and sigmoid, and writes the BF16 hidden row, rounded to nearest-even, to L1.

**Down (72 workers, nine per bank).** Each worker owns one output tile column and loops over the
eight slots, reading 36 K tiles per slot in three blocks of 12. After each slot it rounds the
projection to BF16, multiplies it by that slot's FP32 weight and adds it to an FP32 sum in DEST. It
rounds the sum to BF16 once, after slot 7, and writes the output row to L1.

The data movement is the same in both programs:

- The reader core reads the eight IDs (16 bytes), computes each block's bank-local address from the
  expert ID and issues 13,056-byte reads (12 tiles) into a two-block ring. At most one block of reads
  is in flight per worker while compute consumes the other block. The reader restores the NoC read
  command state when it finishes, because that state survives across launches.
- The writer core gathers the live row of the activation into 8x32 BF16 tiles for the matmul and
  scatters the result rows.
- If `x`, the IDs or the weights are in DRAM, a small staging program copies them to L1 first. The
  fused decode graph produces all three in L1, so a visit runs two programs.

## Speed

In a microbenchmark with the route ready, one visit takes about 146 µs, compared with 234 µs in
the stock path. The 67.7 MB of weights streamed per visit gives 464 GB/s, 95 % of the 490 GB/s a
contiguous DRAM read benchmark reaches. Over 48 visits the experts take about 6.9 ms of the
token. See [design.md](design.md) for the breakdown and [benchmarks.md](benchmarks.md) for the
end-to-end measurements.

Stock indexed `sparse_matmul` also reads only the routed experts, so the gain does not come from
reading fewer bytes. It comes from two things:

- **Layout.** The stock weight is an ordinary interleaved `[1, 128, K, N]` tensor, so consecutive
  tiles of one expert rotate across all eight banks and each worker reads its tiles page by page
  from every bank. The pack gives each reader 13 KB contiguous runs from one nearby bank.
- **Launches.** The stock path is six programs per visit: two output clears, two sparse matmuls, and
  the rowvm activation and combine. The streamer does the activation and the weighted combine inside
  the matmul programs.

The streamer keeps HiFi4. HiFi2 saved about 1.6 % of the visit time and raised the error against an
FP32 reference 1.3-1.8x. BFP4_B experts were rejected for accuracy; see
[precision.md](precision.md).

## Numerics

The streamer reproduces the stock path's rounding points: FP32 accumulation, BF16 at the projection
writeback, FP32 activation and FP32 weighted sum in routing-slot order. On two real layers with
random and real routes it matched every BF16 output value of the stock HiFi4 path.
`inkling_tt/kernels/experts/verify.py` runs this comparison, plus eager-versus-replay checks with
new IDs, weights and inputs at fixed addresses (`scripts/run_tests.sh kernels`).

## Stock fallback

`INKLING_EXPERTS=stock` (read by `inkling_tt/paths.py`) uploads each layer's experts as stock
interleaved tensors and runs them with two indexed `ttnn.sparse_matmul` calls, `rowvm.moe_hidden`
and `rowvm.moe_combine`. The visit-fused decode graph requires the streamer, so this setting also
turns off visit fusion and everything built on it, including the fused attention decode, the fused
router, the head tail and device token feedback. Use it to compare against stock TTNN, not for
serving.

## Prefill

Prefill uses the same pack and the same worker assignment, so the weights are stored once. For a
chunk of T rows (`moe_prefill`, called from `inkling_tt/prefill.py`):

1. Routing runs on device (`route_prefill`): TTNN computes the sigmoid scores and the biased scores,
   one program selects each token's eight experts and weights, and a one-core program builds each
   expert's token list, padded to whole row blocks, and the inverse map.
2. A gather program copies each token's row into compact per-expert blocks of 8 rows (128-row
   chunks) or 32 rows (chunks of 512 rows and more).
3. The gate/up and down programs stream each routed expert's weight slice once per chunk and apply
   it to all of that expert's blocks. An expert with no tokens costs no weight reads. Large chunks
   run several independent lanes of 18 workers each (up to seven, fewer on smaller grids).
4. A combine program restores token order and sums each token's eight weighted terms in FP32 in
   increasing expert-ID order.

The compact 8-row projections keep one 13 KB DRAM read in flight per worker. A worker's next
expert slice stays in the second circular buffer slot while compute consumes the first. Across two
layers with 128 prompt rows, the full MoE kernel chain takes about 2.5 ms per visit.

All workspace shapes depend only on T, so new routes never compile new programs, and the whole path
can be captured in the prefill trace.

## Prior art

The bank-local reader design follows the public [Qwen3-Coder-30B-A3B P150
bundle](https://huggingface.co/raahemnabeel/qwen3-coder-30b-a3b/tree/77c2b34da6ca2591d75bd17cedbefd6c1cbf6a09)
and tt-metal's GPT-OSS fused decode. The design was informed by them; no code was copied.
