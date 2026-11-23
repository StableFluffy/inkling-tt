# inkling-tt

[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on one Tenstorrent Blackhole p150a.

Inkling-2-Mini is a 26B-parameter mixture-of-experts model from Thinking Machines. It loops its middle layers three times, so every token passes through 48 layers. inkling-tt runs it on a single p150a card with TTNN and custom kernels. Dense weights, attention and the KV cache stay in BF16; the routed experts, which the checkpoint ships in NVFP4 and MXFP8, are stored as BFP8_B. Output matches Hugging Face's reference implementation closely, and the repo includes an OpenAI-compatible server.

| | |
|---|---|
| Greedy decode, batch 1 | **18.21 ms/token** (54.9 tokens/s) |
| Time to first token | 0.13 s short prompt, 0.85 s at 1.6k tokens, 3.68 s at 7.8k tokens |
| Context | 32,768 tokens |
| Agreement with HF (BF16, CPU) | 380/384 top-1, 384/384 top-5 next-token predictions |

Decode comes from the traced batch-1 harness and first-token latency from the standalone server on one p150a with all 14 Tensix columns enabled at 380 W. See [docs/benchmarks.md](docs/benchmarks.md) for the measurements and [CHANGELOG.md](CHANGELOG.md) for the release changes.

## Quickstart

Download the converted weights (53 GB) and the official tokenizer files:

```bash
hf download maywell/Inkling-2-Mini-TT-p150a --include 'hf_sd/*' --local-dir .
hf download thinkingmachines/Inkling-2-Mini --revision b85c3cfa615fc413cc9af3379f491b963f3b586a \
  --exclude '*.safetensors' --local-dir ./Inkling-2-Mini
```

The converted weights are the official checkpoint dequantized to BF16 and split per layer. To make them yourself, see [docs/install.md](docs/install.md#3-download-and-convert-the-checkpoint).

Build the release image with the pinned tt-metal tree. See [docs/development.md](docs/development.md#release-image).

```bash
docker run -d --name inkling --privileged \
  --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G \
  -e TT_VISIBLE_DEVICES=0 -e TT_METAL_VISIBLE_DEVICES=0 -p 127.0.0.1:8000:8000 \
  -v "$PWD/hf_sd":/models/hf_sd:ro -v "$PWD/Inkling-2-Mini":/models/Inkling-2-Mini:ro \
  -v "$PWD/cache":/cache \
  inkling-2-mini-tt:0.1.1

until curl -sf localhost:8000/health; do sleep 10; done

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is 2+3?"}], "temperature": 0, "max_tokens": 256}'
```

For greedy DSpark speculation, download the 0.64 GB drafter into the same snapshot directory and append
`--mtp --lmax 8192` after the image name in the command above.

```bash
hf download thinkingmachines/Inkling-2-Mini mtp.safetensors \
  --revision b85c3cfa615fc413cc9af3379f491b963f3b586a --local-dir ./Inkling-2-Mini
```

The first start compiles the kernels into the mounted cache. For tests, benchmarks and source changes, see [docs/install.md](docs/install.md).

## How it works

A decode step is a single captured trace of 607 device programs on the tested grid. Most of the time goes to reading weights. The DRAM-sharded dense projections and a custom expert streamer, which reads the eight routed experts from bank-local packs, run near the card's memory bandwidth. Between the matmuls, chains of small operations (norms, depth mixing, routing, convolutions) are each fused into one custom program, because launching a program costs more than most of these chains compute. Attention, the readout and greedy token feedback also run as fused programs. Sampled decode filters the full vocabulary on the device and reads back one token ID per step.

Read [docs/design.md](docs/design.md) for the time budget and the main techniques.

## Documentation

- [Install from source](docs/install.md)
- [Serving](docs/serving.md): server API, container image, vLLM adapter
- [Hardware](docs/hardware.md): tested configuration, stock cards and worker grids, memory
- [Benchmarks](docs/benchmarks.md): release numbers and how to reproduce them
- [Decode design](docs/design.md), [Prefill](docs/prefill.md), [Precision](docs/precision.md)
- [DSpark draft and verification kernels](docs/speculative.md)
- Kernels: [row epilogues](docs/rowvm.md), [experts](docs/experts.md), [attention](docs/attention.md), [dense projections](docs/dense.md)
- [Model architecture](docs/model.md)
- [Development](docs/development.md): tests, profiling, trace rules, release image

## Limitations

- One sequence at a time; no batching.
- Text only. The vision tower is not ported.
- Tested only on p150a cards with all 14 Tensix columns enabled. Stock cards (12 columns) are supported by the scheduler and emulated on the tested card, but not tested on real hardware.
- Pinned to tt-metal `79951bd4`.
- The server has no authentication.
- The vLLM adapter is experimental and limited to 2,048 tokens.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The Inkling-2-Mini weights are released by Thinking Machines Lab under Apache-2.0, and use of the model is subject to their [Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/). tt-metal is Apache-2.0, Copyright Tenstorrent. The expert-streaming kernels follow the design of the public [Qwen3-Coder-30B-A3B P150 bundle](https://huggingface.co/raahemnabeel/qwen3-coder-30b-a3b/tree/77c2b34da6ca2591d75bd17cedbefd6c1cbf6a09) and tt-metal's GPT-OSS decode; no code was copied from either.
