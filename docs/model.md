# Model

This page describes Inkling-2-Mini as a porter needs it: shapes, the loop schedule, the attention and MoE blocks, and the points where HF eager rounds to BF16. Remember two things. Layers 6-17 run three times, so one token makes 48 layer visits, each with its own KV cache and convolution histories. And a layer does not read a plain residual stream: it reads a learned softmax mix over earlier block states.

The constants are in `inkling_tt/model.py`. The CPU reference is `tools/reference/ref_inkling.py`.

## Dimensions

| Quantity | Value |
|---|---|
| Hidden width `D` | 2304 |
| Physical layers / visits per token | 24 / 48 |
| Query heads x head dim | 24 x 128 |
| KV heads (GQA group 3) | 8 |
| Relative-bias rank / distances | 16 / 1024 |
| Sliding window | 1024 keys |
| Global layers | 5, 11, 17, 23 |
| Routed experts / per token | 128 / 8 |
| Expert width `F` | 1152 |
| Vocabulary (real / stored rows) | 200,058 / 201,024 |
| RMSNorm epsilon | 1e-6 |

The text decoder has about 26 B parameters, almost all of them in the experts. There is no shared expert and no RoPE in the target model.

Below, `RN(x)` is RMSNorm without a weight, `RN(x, w)` with a learned weight `w`, both computed in FP32.

## Loop schedule

```
embed -> L0..L5 -> L6..L17 (pass 0) -> L6..L17 (pass 1) -> L6..L17 (pass 2) -> L18..L23 -> readout
                   \_______ each pass restarts from the state after L5 _______/
```

The core layers share weights across passes, but each visit keeps its own K/V cache and four convolution histories. `visit_slot(layer, pass)` in `inkling_tt/model.py` maps a visit to one of 48 cache slots: the layer index on pass 0 and outside the core, `24 + 12*(pass-1) + (layer-6)` on passes 1 and 2. Of the 48 visits, 8 are global (layer 5, layers 11 and 17 three times each, layer 23) and 40 are sliding.

Only the caches and convolution histories carry across tokens. The block bank and the cross-pass messages below are rebuilt inside every forward.

## Depth mixing

The model keeps a running partial state `x` and a bank `C` of completed block states. Blocks start at layers 0, 6, 12 and 18. At a block start the incoming partial is normalized and appended to the bank, and the partial restarts from the layer's attention output. Inside a block the partial accumulates as a normal residual.

A layer reads its input through a depth mix with learned query rows `Q`:

```
s    = [c_1, ..., c_J, RN(x)]                   bank entries are stored normalized
p    = softmax_j(q . s_j / sqrt(2304))          FP32, one softmax per token and query row
Mix  = sum_j p_j s_j
```

One layer visit, with `q_attn` and `q_mlp` the layer's learned query rows:

```
u       = Mix(C, x, q_attn)                     layer 0 uses x directly
xq, xk, xv = RN(u, w_in)                        BF16
a       = conv_post_attn(Attention(xq, xk, xv))
if block start:  C = C + [RN(x)];  z = a
else:            z = x + a
x'      = BF16(z + conv_post_mlp(MoE(RN(Mix(C, z, q_mlp), w_post))))
```

Sliding layers use one query row and one input norm for Q, K and V. Global layers use three query rows, giving separate Q, K and V streams, each with its own input norm. The readout mixes the four bank entries and the final partial with the output query, applies the final norm and the LM head.

## Message gates

On passes 1 and 2 each core layer also receives messages from the previous pass. The message bank holds that pass's block appended at layer 12 and its final partial after layer 17. The block appended at layer 6 duplicates the pre-loop state and is skipped. Core layer `6+j` mixes this bank with four learned query rows (`message_input.query[j]`) into messages for Q, K, V and the MLP read. Pass 0 has no messages.

A controller blends each mixed read `u` with its message `m` before the learned input norm:

```
h  = W_down . BF16([RN(u); RN(m)])              64 wide, BF16 projection
f  = tanh(W_f h),  g = tanh(W_g h)              FP32 weights and math
u' = (1 - f) * u + g * m
```

Both gates are signed, so the factor on `u` can exceed 1. Sliding layers share one mixed read but still run separate Q, K and V controllers, so their three attention inputs differ on repeat passes.

## Attention

From `xq` the layer projects Q `[24,128]`, the relative query R `[24,16]`, one sink logit per head `[24]` and an output gate `[3072]`. K and V `[8,128]` come from `xk` and `xv`. No projection has a bias.

K and V pass through a four-tap causal depthwise convolution before normalization. Sliding layers use static taps; global layers add input-dependent taps.

```
static:   y_t = (x_t + sum_{j=0..3} w_j x_{t+j-3}) / sqrt(5)              tap 3 is the current row
dynamic:  y_t = (x_t + sum_{l=0..3} (w_l + d_{t,l}) x_{t-l}) / 3          lag 0 is the current row
          w: learned static taps; d = W_taps . xk (or xv), 4 lags per KV head,
          shared by the head's 128 channels
```

Then each head is normalized:

```
clip(y, c) = y * min(1, c / (||y||_2 + 1e-6))
Q = clip(sigma_q * Q, 2*sqrt(128))     R = clip(sigma_r * R, 8)
K = clip(K, 2*sqrt(128))               V = RN(V) per head
global layers only:  Q, R *= tau(p) = 1 + 0.1 * ln(max((p+1)/8192, 1))
```

`sigma_q` and `sigma_r` are learned per-head, per-channel scales. The cache stores these final K and V rows.

Scores add a learned relative bias after the QK scale. Query head `h` reads KV head `h // 3`.

```
score(h, k) = Q_h . K_k / sqrt(128) + R_h . P[:, delta]      delta = p_q - p_k
P is a learned [16, 1024] table; the bias is zero for delta >= 1024
```

Sliding layers see `0 <= delta < 1024`, global layers every earlier key. The sink logit joins the softmax denominator per query and head, and its probability is dropped; equivalently, ordinary attention output times `sigmoid(logsumexp(scores) - sink)`. The 24 head outputs are multiplied by `sigmoid(gate)`, projected by `W_o` and passed through the post-attention convolution.

Each visit keeps four convolution histories for the K and V projections (1024 wide each), the attention output and the MoE output (2304 wide each). A convolution reads the three preceding inputs. The fused decode layout indexes the histories by position as described in [rowvm.md](rowvm.md).

## MoE

```
s     = sigmoid(W_router . x)                    FP32, all 128 experts
ids   = top8(s + b)                              the bias b affects selection only
gamma = 8 * s[ids] / (sum s[ids] + 1e-20)
[g; u] = W_13 . x                                1152 + 1152
h     = 4 tanh(g/4) * sigmoid(g) * 25 tanh(u/25) the sigmoid sees the raw gate
MoE   = sum_e gamma_e * (W_2 . BF16(h_e))        FP32 sum
```

In the converted shards `gate_up_proj` is `[128, 2304, 2304]` with gate rows first and up rows second; `down_proj` is `[128, 2304, 1152]`.

## Readout

`logits = 30 * tanh(W_head . RN(Mix(C, x, q_out), w_norm) / 30)`. The head is not tied to the embedding. The checkpoint stores 201,024 rows for both; rows from 200,058 up are padding. Transformers drops the padded head rows, so the converted head has 200,058 rows, and the port masks the tile padding of its logits before argmax. The embedding row is normalized with `RN(e, w_embed)` and has no extra scale.

## Cast points

To match HF eager, round to BF16 exactly where it does and keep everything else in FP32:

- the normalized embedding, and every projection input after its learned input norm;
- every projection output, including the gate/up and down outputs of each expert;
- Q and R after clipping, and again after the `tau` multiply on global layers;
- K and V after the convolution, and again after clip or RMS, as stored in the cache;
- attention probabilities before the product with V, and the gated output before `W_o`;
- each branch convolution output before the FP32 residual add, and the layer output;
- the expert hidden `h` before `W_2`, and the combined MoE output;
- the controller input concatenation and its 64-wide projection;
- the LM head output, before the FP32 softcap.

Depth mixing, norms, clips, `tau`, convolution accumulation, controller gates, softmax statistics, the router, expert activations, the expert combine and the softcaps are FP32. [precision.md](precision.md) lists where the port departs from this.

Three details are easy to get wrong. The clip epsilon is added after the square root, the RMSNorm epsilon inside it. The relative bias and the sink are not multiplied by the 1/sqrt(128) score scale. Masking padded vocabulary before the softcap turns `-inf` into `-30`, which makes padding tokens sampleable.

Native engines differ from HF eager: vLLM and SGLang quantize expert activations to NVFP4 or MXFP8 and keep FP32 logits. Compare intermediate tensors against HF eager or the reference, not against those engines.

## Checkpoint

The published experts are NVFP4 on layers 0-19 and MXFP8 on layers 20-23; all other text tensors are BF16 or FP32. `tools/convert/convert_checkpoint.py` loads the model through Transformers, which dequantizes the experts to BF16 and splits the interleaved gate/up rows, and writes one shard per layer. The port packs the experts to BFP8_B at load. The vision tower and the MTP draft head are not part of the text decoder.

## Reference implementations

`tools/reference/ref_inkling.py` is a CPU reference that follows HF eager arithmetic. It reads the original checkpoint names from a local snapshot, dequantizes only the selected experts, and returns FP32 logits with an explicit cache. A `trace(visit, physical_layer, pass_idx, partial)` callback exposes every post-layer partial. `--hf-loader-rounding` reproduces the Transformers generic loader, which rounds the NVFP4 per-expert scale to BF16 before dequantizing.

```sh
python tools/reference/ref_inkling.py "$INKLING_SNAPSHOT" 200001 42 43 --steps 3
```

`InklingTT` in `inkling_tt/model.py` is the first device port: one TTNN op at a time, the cast points above, routing on the host. It is slow but reads as an executable specification. The traced decode in `inkling_tt/model2.py` and the prefill in `inkling_tt/prefill.py` compute the same model with fused kernels.
