# Hardware

inkling-tt runs on one Tenstorrent Blackhole p150a. All published numbers come from cards with all 14 Tensix columns enabled and a 380 W board power limit. Stock cards expose fewer columns; the code supports them, but no physical stock card has been tested.

## Tested configuration

| | |
|---|---|
| Card | 1x p150a, 32 GB GDDR6 in 8 DRAM banks, AICLK 1.35 GHz |
| Worker grid | 13x10, plus one column for dispatch |
| Board power limit | 380 W |
| Firmware, driver | firmware bundle 19.17.1, tt-kmd 2.12.0 |
| tt-metal | `79951bd44b7484a136b3619e0d81f26baa8a1936`, Release build |
| Dev image | `tt-metalium/ubuntu-22.04-dev-amd64@sha256:47772e38…` |
| Host | x86-64 Linux, Docker, 1 GiB hugepages |

The dev image pins Python 3.10, torch 2.11.0+cpu and Transformers 5.12.1, which the runtime uses only for the tokenizer and chat template. The checkpoint is `thinkingmachines/Inkling-2-Mini` at revision `b85c3cfa`.

Other tt-metal commits are untested. The custom kernels call tt-metal device APIs directly, so expect to adapt them when moving to another release.

## Worker grids

A stock p150a has two harvested Tensix columns and reports an 11x10 worker grid. Cards with one harvested column report 12x10, and fully enabled cards 13x10. The model reads the grid with `compute_with_storage_grid_size()` and schedules every custom program for it (`inkling_tt/grid.py`):

- Row epilogues, the router, the head tail and the dense projections size their core grids from the reported width.
- Fused attention runs all eight KV heads in one launch on 13x10 and in two launches of four heads on smaller grids. Each head keeps its split and reduction order.
- The expert streamer, paired K/V and prefill projections place their readers near each bank using tt-metal's DRAM-bank assignment.
- Grids smaller than 11x10, or cards without eight DRAM banks, are rejected before the device opens.

Changing the grid changes scheduling, not results. On the tested card, restricted to 11x10 and 12x10, every parity and evaluation fixture produced bit-identical logits to the 13x10 run. Decode was about 2.5 % slower at 11x10 and the 8k-token prefill about 8 % slower.

## Emulating a smaller grid

`INK_GRID` restricts all scheduling, including TTNN ops that choose their own cores, to a smaller grid:

```bash
INK_GRID=11x10 scripts/run_tests.sh parity
```

Set it before the process opens the device, and leave it unset on a stock card. It only limits which workers run programs. The card keeps its own harvesting pattern, NoC routing, clocks and power limit, so stock-card speed is not measured by this. The program placement was also checked against the stock harvesting descriptor on a tt-metal mock device, without running kernels.

## Memory

At `lmax` 32768, with the decode trace, both prefill buckets and the on-device embedding table resident, tt-metal reports 31.35 GB of DRAM allocated out of 33.78 GB. A shorter `lmax` shrinks the KV caches of the eight global-attention visits. The 40 sliding-window visits use fixed 1024-slot rings.

With both prefill buckets resident, the larger dispatch ring leaves 1,105,216 contiguous L1 bytes per worker. Wide global attention uses about 1.1 MB of that space for its circular buffers.
