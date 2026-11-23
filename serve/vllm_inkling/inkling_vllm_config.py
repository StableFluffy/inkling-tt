# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from src/transformers/models/inkling2_loop/configuration_inkling2_loop.py in
# https://github.com/huggingface/transformers at be48e7a0b63d2ff22ce1ebccbccbd0374c1592ef.
# Modified by the inkling-tt authors (2026): standalone module for transformers 5.12 (no relative imports,
# @strict and docstring decorators removed, loop_config folded in __post_init__) plus AutoConfig registration.

import contextlib

from transformers import AutoConfig, PreTrainedConfig


class Inkling2LoopTextConfig(PreTrainedConfig):
    model_type = "inkling2_loop_text"
    base_config_key = "text_config"
    attribute_map = {
        "sliding_window": "sliding_window_size",
        "num_local_experts": "n_routed_experts",
        "sconv_kernel_size": "conv_kernel_size",
        "model_max_length": "max_position_embeddings",
        "intermediate_size": "moe_intermediate_size",
        "routed_scaling_factor": "route_scale",
    }

    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    hidden_size: int = 2304
    num_hidden_layers: int = 24
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    swa_num_attention_heads: int = 24
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 128
    sliding_window_size: int = 1024
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: int = 8192
    log_scaling_alpha: float = 0.1
    layer_types: list[str] | None = None
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    conv_kernel_size: int = 4
    hidden_act: str = "silu"
    moe_intermediate_size: int = 1152
    n_routed_experts: int = 128
    num_experts_per_tok: int = 8
    route_scale: float = 8.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = 200006
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    n_layers_per_attn_res_block: int = 6
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    final_logit_softcapping: float = 30.0
    scoring_func: str = "sigmoid"
    use_cache: bool = True
    loop_layer_index_start: int = 6
    loop_layer_index_end: int = 18
    loop_num_passes: int = 3
    loop_gate_rank: int = 64

    def __post_init__(self, **kwargs):
        loop = kwargs.pop("loop_config", None)
        if loop is not None:
            self.loop_layer_index_start = loop.get("core_start", self.loop_layer_index_start)
            self.loop_layer_index_end = loop.get("core_end", self.loop_layer_index_end)
            self.loop_num_passes = loop.get("num_passes", self.loop_num_passes)
            self.loop_gate_rank = loop.get("gate_rank", self.loop_gate_rank)
        if self.layer_types is None:
            self.layer_types = [
                "hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding" for i in range(self.num_hidden_layers)
            ]
        super().__post_init__(**kwargs)


class Inkling2LoopConfig(PreTrainedConfig):
    model_type = "inkling2_loop"
    sub_configs = {"text_config": Inkling2LoopTextConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | None = None
    mtp_config: dict | None = None
    image_token_id: int = 200054

    def __post_init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        if isinstance(self.text_config, dict):
            self.text_config = Inkling2LoopTextConfig(**self.text_config)
        elif self.text_config is None:
            self.text_config = Inkling2LoopTextConfig()
        super().__post_init__(**kwargs)


ARCH = "Inkling2LoopForConditionalGeneration"
ADAPTER = "inkling_vllm_adapter:InklingForCausalLM"


def register():
    from vllm.model_executor.models.registry import ModelRegistry

    for name in (ARCH, "TT" + ARCH):
        if name not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(name, ADAPTER)
    for name, cls in (("inkling2_loop", Inkling2LoopConfig), ("inkling2_loop_text", Inkling2LoopTextConfig)):
        with contextlib.suppress(ValueError):
            AutoConfig.register(name, cls)
