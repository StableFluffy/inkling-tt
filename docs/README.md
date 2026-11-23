# Documentation

Start with the top-level [README](../README.md).

Using inkling-tt:

- [install.md](install.md): build tt-metal, convert the checkpoint, run the tests
- [serving.md](serving.md): the HTTP server, the container image and the vLLM adapter
- [hardware.md](hardware.md): tested configuration, stock cards and worker grids, memory
- [benchmarks.md](benchmarks.md): release measurements and how to reproduce them

How it works:

- [design.md](design.md): the decode step, where its time goes and the techniques behind it
- [prefill.md](prefill.md): traced chunked prefill
- [speculative.md](speculative.md): the DSpark drafter and greedy target verification
- [precision.md](precision.md): the precision policy and the evidence for it
- [model.md](model.md): the Inkling-2-Mini architecture, as needed to port it

Kernels:

- [rowvm.md](rowvm.md): fused row epilogues between matmuls
- [experts.md](experts.md): streaming the routed experts
- [attention.md](attention.md): fused attention decode
- [dense.md](dense.md): DRAM-sharded dense projections

Working on the code: [development.md](development.md) covers the environment, test tiers, profiling, trace rules and the release image.
