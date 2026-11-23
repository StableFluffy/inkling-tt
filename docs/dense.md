# Dense decode projections

At batch 1 every dense projection is a matrix-vector product: one row of activations against a
weight that must be read from DRAM in full. `inkling_tt/dense.py` gives each projection shape a
tuned matmul configuration, chosen from measurements, and stores the weight in the layout that
configuration reads fastest. The large projections reach about 90 % of the card's 512 GB/s; the
small ones are bound by launch cost, not bandwidth.

## Configurations

`prepare_weight(device, w, dtype)` looks up `(K, N, dtype)` in `_CONFIGS`, uploads the `[K, N]`
weight and returns a `DenseWeight` handle with the program config, compute config and shard layouts.
Shapes without an entry are rejected. The stock matmuls use two kinds of configuration:

- **DRAM-sharded pipeline** (`pipe`). The weight is width-sharded across the eight DRAM banks, `32 *
  ceil(N / 256)` columns per bank. The activation is width-sharded in L1, and the output is
  width-sharded in L1. tt-metal's `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` with
  `use_multicore_pipeline=True` places its compute workers next to the bank they read; with several
  workers per bank it also splits K inside the bank.
- **1D multicast** (`1d`). The weight is interleaved in DRAM and
  `MatmulMultiCoreReuseMultiCast1DProgramConfig` multicasts the activation across the worker grid,
  one output tile per core. For narrow outputs this measured faster than DRAM sharding.

The decode graph uses these shapes, all with BF16 weights at HiFi4 and FP32 accumulation:

| Projection | K x N | Config | Workers per bank | K block | Input / output shard tiles |
|---|---|---|---:|---:|---|
| Q, relative, sink, gate (fused) | 2304 x 6560 | pipe | 4 | 2 | 4 / 4 |
| O | 3072 x 2304 | pipe | 4 | 2 | 4 / 2 |
| LM head | 2304 x 201024 | pipe | 4 | 2 | 4 / 61 |
| K and V, one program | 2304 x 1024 each | bank-local | 8 | 8 | |
| Dynamic conv taps (global) | 2304 x 32 | 1d | | 24 | |
| Router | 2304 x 128 | 1d | | 36 | |
| Message-gate down | 4608 x 64 | pipe | 8 | 6 | 4 / 1 |

In the fused decode graph two of these handles serve as weight storage for rowvm programs rather
than as matmuls: `rowvm.fused_router` reads the interleaved router weight directly, and
`rowvm.message_down` reads its own tile-interleaved copy of the message-gate down weight. Prefill
uses the DRAM-sharded message-gate down handle. `_CONFIGS` also holds entries for BFP8_B weights
(HiFi2) and for the separate Q, relative, sink and gate shapes; the release graph does not use them.

## Calling a projection

`dense.linear(x, handle)` takes an interleaved TILE activation of shape `[1, 1, 1, K]` in L1 or
DRAM. It casts an FP32 activation to BF16, moves it into the input shards if the config needs them,
runs the matmul and returns an interleaved row in the input's memory. Two shapes have fixed output
types: the router returns FP32, and the message-gate down projection rounds to BF16 and returns it
promoted to FP32, as the HF model does.

The decode graph avoids the conversions where it can. `model2._linear_to_shards` keeps the native
output shards of the fused Q projection, the O projection and the LM head, because their consumers
read shards directly: `rowvm.prepare_attention` reads Q, R, the sink and the gate logits from the
packed output, the attention kernel writes the O projection's input shards, and `rowvm.head_tail`
reads the head's output shards.

B1 and MTP use the same four-worker BF16 head schedule. MTP shares the projection
workspace across prefill, decode and draft.

## The fused attention projection

The four projections of the attention input share one weight, packed by `prepare_fused_weight` in
the order Q (3072), relative R (384), sink (24, padded to 32) and gate (3072). The model folds the
learned Q and R clip scales and the `sqrt(128)` sink factor into the weights at load time. Consumers
read the four parts as column windows at offsets 0, 3072, 3456 and 3488 of the 6560-wide output.

One fused matmul reads the activation once and is one launch instead of four. With the input shard
conversion shared in both cases, the fused projection measured about 1.09x faster than four separate
ones.

## Paired K and V

Decode runs K and V together with their own BF16 inputs. Each of 64 workers owns one output tile
and streams its column of weights from one bank. The compute kernel keeps the same eight-tile
K blocks and FP32 accumulation order as the 1D projections. The resulting rows stay in two
persistent L1 buffers for convolution, normalization and the cache write.

The pair takes about 25 µs in isolation, compared with 27 µs for the two 1D matmuls. On a 13x10
grid it shares a launch with head preparation and the cache writers, saving a further 48 launches
per token. The model keeps the original interleaved weights for prefill and uses about 226 MB
of additional DRAM for the bank-local decode copies. The cache handoff is described in
[attention.md](attention.md).

## Bandwidth

Measured in isolation on the tested card, BF16 weights, HiFi4, weight bytes divided by warmed matmul time:

| Projection | GB/s | Share of 512 GB/s |
|---|---:|---:|
| Q, relative, sink, gate (fused) | 470 | 92 % |
| LM head | 489 | 95 % |
| O | 451 | 88 % |
| Paired K and V | 384 | 75 % |
| Router | 74 | 15 % |
| Dynamic conv taps | 26 | 5 % |

The router and tap weights are 0.6 MB and 0.15 MB, so a few microseconds of launch and pipeline fill
dominate their time. The decode step's weight-streaming programs, including the experts, run close
to the rate the card sustains; [design.md](design.md) has the per-token breakdown and
[benchmarks.md](benchmarks.md) the release figures.

The router weight stays in the 1D layout. A DRAM-sharded split-K config had a shorter matmul kernel,
but its K-group reduction can round FP32 partials, and in an exactness test it changed most router
outputs; small changes there can flip the top-8 selection.

## Worker grid

The tile-level parameters in `_CONFIGS` are the same on every supported grid (11x10 to 13x10,
[hardware.md](hardware.md)). What depends on the grid is where the work lands:

- `prepare_weight` calls `grid.worker_grid(device)`, which rejects grids below 11x10, and requires
  eight DRAM banks.
- 1D configs use the reported worker grid as their compute grid.
- Activation and output shards are laid out row-wise over the reported grid with
  `num_cores_to_corerangeset`. The fused Q projection's 52 output shards fill four rows of a 13-wide
  grid and five rows of an 11-wide one. The LM head needs 103 output cores, which fits the 110
  workers of an 11x10 grid.
- The DRAM-sharded pipeline chooses its compute workers from the device's DRAM-bank affinity.
  `INK_GRID` sets tt-metal's worker-grid limit before the device opens, so this factory also stays
  inside a restricted grid.

On the tested card restricted with `INK_GRID`, the parity fixtures produce identical logits at
11x10, 12x10 and 13x10.

## Prefill

Prefill reuses the decode weights. `linear_prefill_resident` runs a custom program that reads each
DRAM-sharded weight tile once per chunk and keeps it in L1 for all rows of the chunk, with the same
bank-adjacent worker assignment and two-tile K blocks for Q and O.
