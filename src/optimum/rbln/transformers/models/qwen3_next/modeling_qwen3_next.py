# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from transformers import PretrainedConfig

from ...models.decoderonly import RBLNDecoderOnlyModelForCausalLM
from .qwen3_next_architecture import Qwen3NextWrapper


class RBLNQwen3NextForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    RBLN wrapper for HuggingFace `Qwen3NextForCausalLM`.
    Supports both full-attention and linear-attention layers.
    """

    _decoder_wrapper_cls = Qwen3NextWrapper

    @classmethod
    def get_input_info(
        cls,
        batch_size: int,
        query_length: int,
        rbln_config,
        model_config: PretrainedConfig,
    ):
        input_info = super().get_input_info(batch_size, query_length, rbln_config, model_config)

        layer_types = getattr(model_config, "layer_types", None) or []
        linear_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
        kv_layers = [i for i, t in enumerate(layer_types) if t != "linear_attention"]

        kv_keep = set()
        for layer_idx in kv_layers:
            kv_keep.add(f"past_key_values_{layer_idx * 2}")
            kv_keep.add(f"past_key_values_{layer_idx * 2 + 1}")

        filtered = []
        for name, shape, dtype in input_info:
            if name.startswith("past_key_values_") and name not in kv_keep:
                continue
            filtered.append((name, shape, dtype))
        input_info = filtered

        if len(linear_layers) == 0:
            return input_info

        # state_cache shape (blocked layout):
        #   [num_blocks, H, Dk/64, 64, Dv/64, 64]

        num_blocks = rbln_config.kvcache_num_blocks
        H = model_config.linear_num_value_heads
        Dk = model_config.linear_key_head_dim
        Dv = model_config.linear_value_head_dim
        Dk_blk = Dk // 64
        Dv_blk = Dv // 64

        for layer_idx in linear_layers:
            input_info.append(
                (
                    f"past_key_values_linear_{layer_idx}",
                    [num_blocks, H, Dk_blk, 64, Dv_blk, 64],
                    rbln_config.dtype,
                )
            )

        return input_info
