# Changelog

## 0.1.1

- Short-prompt batch-1 greedy decode drops from 20.0 to 18.2 ms/token. Paired K/V projection and cache preparation share a launch, and a larger dispatch ring cuts time between programs.
- Warm prefill TTFT at 128, 300, 1,024, 2,048, 8,192 and 32,768 tokens is 0.169, 0.463, 0.710, 0.892, 3.745 and 18.395 s. The v0.1.0 figures were 0.207, 0.567, 0.852, 1.036, 4.313 and 22.88 s.
- Sampled decode applies temperature, penalties, top-k and top-p across the full vocabulary on the device and returns one ID per step. Repeating a seed reproduces the stream on the same sampling path.
- Greedy DSpark speculation is opt-in with `--mtp` and the official `mtp.safetensors` shard. Across seven prompts the mean decode median falls from 18.84 to 17.83 ms/ID in the same server. Python falls from 18.43 to 16.12, math from 18.49 to 15.78, and the 7,820-token case rises from 20.02 to 20.10 ms/ID. See [docs/benchmarks.md](docs/benchmarks.md#adaptive-greedy-mtp).
