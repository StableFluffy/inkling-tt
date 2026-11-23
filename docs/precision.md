# Precision

The port follows HF eager arithmetic everywhere except expert weight storage. Dense weights, the LM head and the KV caches are BF16, the routed experts are BFP8_B, every matmul runs at HiFi4 with FP32 accumulation, and reductions and nonlinearities run in FP32 with the accurate SFPU functions. The decisive number: BFP8_B experts add about 1 % error to a MoE layer's output, BFP4_B experts about 17-21 %, so the experts stay at 8 bits.

## Policy

| Part | Storage | Math |
|---|---|---|
| Dense projections (Q bundle, K, V, O, dynamic taps, controller down) | BF16 | HiFi4, FP32 accumulation |
| Router | BF16 | FP32 accumulation, FP32 selection |
| Routed experts | BFP8_B | HiFi4, FP32 accumulation |
| Controller up projections | FP32 | FPU matmul, FP32 accumulation |
| LM head | BF16 | HiFi4, FP32 accumulation, BF16 output |
| K/V caches | BF16 | |
| Convolution histories | FP32 | |
| Norms, clips, `tau`, depth mix, conv taps, softmax, softcaps, expert combine | | FP32, accurate SFPU |

`Precision` in `inkling_tt/model.py` carries `expert_dtype`, `dense_dtype`, `expert_fidelity` and `dense_fidelity`. `default_precision()` in `serve/runtime.py` sets them to BFP8_B, BF16, HiFi4 and HiFi4 and selects the streaming expert path; the tests use the same settings. An optional `head_dtype` attribute overrides the LM head's dtype and defaults to `dense_dtype`.

Two fields reach less than their names suggest. The streaming expert path (`INKLING_EXPERTS=stream`, the default) always packs BFP8_B and runs HiFi4; `expert_dtype` and `expert_fidelity` only reach the stock `sparse_matmul` path. Visit fusion in the decode graph requires `dense_dtype` BF16; any other dense dtype runs the unfused graph.

## Matmul settings

With the default settings, every compute configuration in the port (dense projections, custom kernels and the stock ops it configures) uses `math_fidelity=HiFi4` and `fp32_dest_acc_en=True`, with `math_approx_mode` off. HiFi4 runs all four fidelity phases, so a product of two BF16 or BFP8_B operands keeps every operand bit; FP32 Dest keeps the sum in FP32 until the kernel packs its output. `dense.py` also holds BFP8_B configurations at HiFi2, used only when `dense_dtype` is BFP8_B.

The FPU's source registers hold at most TF32 precision, so a matmul with FP32 operands is TF32-precision even at HiFi4. FP32 work that must stay exact therefore runs on the SFPU, with FP32 operands unpacked straight into Dest (`UnpackToDestFp32`); the decode depth mix and the router selection work this way. Two products still use the FPU with FP32 operands: the controller up projections (FP32 in HF) and, in prefill only, the depth-mix logits.

Where a kernel rounds FP32 to BF16 in software at an HF boundary, it rounds to nearest even (`float32_to_bf16_rne`); Blackhole's own conversion resolves ties away from zero.

## Transcendentals

The custom kernels use the non-approximate SFPU paths. Sigmoid, tanh and square-root initialize in accurate mode. Fused attention decode uses the accurate FP32 exponential (`_sfpu_exp_fp32_accurate_`); prefill attention uses `exp_tile<false>`. The decode expert kernels compute the softcaps with the accurate FP32 tanh; the prefill expert kernels use tt-metal's SFPU softcap. In rowvm and fused attention, reciprocals start from the hardware estimate and refine it with two Newton steps. The stock SDPA decode fallback runs with `exp_approx_mode=False`. A few prefill steps still call stock TTNN ops at their default settings: the router sigmoid, the attention-gate sigmoid and the depth-mix softmax.

## Expert format

TT block-float formats store one shared 8-bit exponent per 16 consecutive values of a tile face row. BFP8_B keeps a sign and 7 magnitude bits per value (1,088 bytes per 32x32 tile), BFP4_B a sign and 3 magnitude bits (576 bytes). The port packs the experts from the converted BF16 shards at load (`experts_stream.prepare`), so its BFP8_B values are rounded twice, through BF16; the study below packed from FP32. All 24 layers in BFP8_B take 26.0 GB; BF16 experts would take 48.9 GB and do not fit on a 32 GB card.

The study behind the choice ran on the CPU. For six checkpoint layers it decoded the experts to FP32, packed them with an emulator of tt-metal's BFP encoder, and pushed 64 seeded Gaussian vectors through the layer's post-attention norm and MoE with HF's routing held fixed, FP32 accumulation and HF's BF16 cast points. The table gives the relative L2 error of the MoE output against the same computation with the checkpoint's exact weights, rounded once to FP32.

| Layer | Source format | BF16 | BFP8_B | BFP4_B |
|---:|---|---:|---:|---:|
| 0 | NVFP4 | 0.53 % | 1.23 % | 18.7 % |
| 5 | NVFP4 | 0.48 % | 1.13 % | 17.3 % |
| 10 | NVFP4 | 0.51 % | 1.20 % | 18.5 % |
| 19 | NVFP4 | 0.54 % | 1.29 % | 19.9 % |
| 20 | MXFP8 | 0.00 % | 0.77 % | 20.8 % |
| 23 | MXFP8 | 0.00 % | 0.76 % | 20.3 % |

The BFP columns share exponents along output channels. Sharing them along input channels instead moves BFP8_B by at most 0.02 points and BFP4_B by at most 0.14, giving a BFP4_B range of 17.2-21.0 % across both. MXFP8 values are exact in BF16, hence the zero column. NVFP4 values are not, so HF's own BF16 weights already carry about 0.5 %; BFP8_B roughly doubles that. BFP4_B is more than ten times worse again and is rejected.

## Fidelity

HiFi2 drops the low fidelity phases. On the expert stream with real layer 0 and 20 weights and routes, it raised the error against an FP32 PyTorch reference by 1.3-1.8x (layer 0: 1.22-1.27 % to 1.60-1.67 %; layer 20: 0.69-0.74 % to 1.18-1.27 %) and saved about 2 us, roughly 1.6 %, of a visit's expert time. The experts stay at HiFi4. A CPU model of the fidelity phases puts LoFi at 5.1-5.5 % on the NVFP4 layers.

## Departures from HF eager

The port deviates from the cast points in [model.md](model.md#cast-points) in these places:

- Experts are BFP8_B instead of BF16 (above).
- The learned Q and R clip scales are folded into the BF16 projection weights at load. HF applies them in FP32 after a BF16 projection.
- The sink weights and the relative-bias table are multiplied by sqrt(128) at load, in BF16, so that the attention kernels can apply one 1/sqrt(128) scale to all logits.
- The router and the projections accumulate in a different order from PyTorch: in K stripes, in 96-wide K blocks (prefill router) or along the DRAM-sharded decode schedule.
- The controller up projections and the prefill depth-mix logits are TF32-precision (above).
- Decode stores normalized bank rows with the low 14 mantissa bits cleared, matching the stock selector matmul that the fused kernel replaced.

Fused attention decode keeps scores and softmax statistics in FP32 and rounds the probabilities and the gated output to BF16. Prefill attention rounds its probabilities to BF16 the same way. Embedding normalization runs on the host in FP32 and rounds to BF16, as in HF.

## Measuring agreement

Accuracy here means agreement with HF eager in BF16 on the CPU, on local fixtures. It is not a task benchmark.

- `tools/golden/hf_eval.py` runs three chat prompts (an explanation, a Python function, a Korean question) through HF eager and stores up to 128 greedy tokens with their logits in FP16. `tests/eval_tf.py` prefills each prompt with the 128-row bucket and then feeds HF's tokens, not its own, so every position compares the same context. It counts top-1 (device argmax equals HF argmax), top-5 (device argmax is in HF's top five) and the relative L2 error of the softcapped logits. It passes at 97 % top-1, 99 % top-5 and a mean relative error of 0.05 or less.
- `tests/test_long.py` runs a 300-token prompt from `tools/golden/hf_long.py`. The last prefill logits must be within 0.04 relative error of HF, and with HF's tokens fed back, the device argmax must match HF's at all but two positions.
- `tests/test_prefill_buckets.py` checks that bucket splits give identical logits and repeats the 300-token check at `lmax` 32768.

Release figures are in [benchmarks.md](benchmarks.md).

## Reproducing the study

The scripts in `tools/precision/` run on the CPU against the original checkpoint (`INKLING_SNAPSHOT`) in the CPU environment from [install.md](install.md). `verify_bfp.py` compiles tt-metal's block-float encoder from `TT_METAL_HOME` and checks the Python emulator against it bit for bit.

```sh
cd tools/precision
P="$INKLING_WORK/cpu-venv/bin/python"
$P verify_bfp.py --build-fast fast_bfp.so            # emulator vs tt-metal encoder; builds the fast packer
$P study.py --threads 4 --fast-bfp fast_bfp.so       # storage and MoE errors, writes results.json
$P tables.py moe                                     # the MoE table above
```

`dequant.py <layer>` decodes and validates one layer, `inventory_loss.py` counts BF16-inexact expert values over all layers, and `budget.py` computes the capacity figures.
