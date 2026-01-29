# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ..decoderonly.decoderonly_architecture import DecoderOnlyWrapper, apply_rotary_pos_emb
from .configuration_qwen3_vl import RBLNQwen3VLVisionModelConfig


class Qwen3VLVisionModelWrapper(nn.Module):
    def __init__(self, model: torch.nn.Module, rbln_config: RBLNQwen3VLVisionModelConfig):
        super().__init__()
        self.merger = model.merger
        self.rbln_config = rbln_config
        self.blocks = self.wrap_vision_blocks(model.blocks, rbln_config)
        self.deepstack_visual_indexes = model.deepstack_visual_indexes
        self.deepstack_merger_list = model.deepstack_merger_list

    def wrap_vision_blocks(
        self,
        blocks: torch.nn.ModuleList,
        rbln_config: RBLNQwen3VLVisionModelConfig,
    ):
        wrapped_blocks = []
        for block in blocks:
            wrapped_blocks.append(Qwen3VLVisionBlock(block, rbln_config))
        return nn.ModuleList(wrapped_blocks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        # Convert attention mask: 1 -> 0, 0 -> -inf
        attn_mask = (1.0 - attn_mask) * torch.finfo(hidden_states.dtype).min

        deepstack_features = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, attn_mask, [cos, sin])
            if layer_num in self.deepstack_visual_indexes:
                deepstack_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[deepstack_idx](hidden_states)
                deepstack_features.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)
        return hidden_states, *deepstack_features


class Qwen3VLVisionBlock(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        rbln_config: RBLNQwen3VLVisionModelConfig,
    ):
        super().__init__()
        self._origin_model = model
        self.rbln_config = rbln_config
        self.norm1 = model.norm1
        self.norm2 = model.norm2
        self.attn = Qwen3VLVisionAttention(model.attn, rbln_config)
        self.mlp = model.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attn_mask,
            position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, model: nn.Module, rbln_config: RBLNQwen3VLVisionModelConfig) -> None:
        super().__init__()
        self._origin_model = model
        self.rbln_config = rbln_config
        self.num_heads = model.num_heads
        self.head_dim = getattr(model, "head_dim", model.proj.in_features // model.num_heads)
        self.qkv = model.qkv
        self.proj = model.proj
        self.scale = torch.tensor(1 / math.sqrt(self.head_dim), dtype=rbln_config.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        hidden_states = hidden_states.unsqueeze(0)
        q, k, v = (
            self.qkv(hidden_states).reshape(1, seq_length, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        )

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn_weights = attn_weights + attn_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=hidden_states.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(1, seq_length, -1)
        attn_output = self.proj(attn_output).squeeze(0)

        return attn_output


class Qwen3VL_LanguageModelWrapper(DecoderOnlyWrapper):
    def get_decoder_layers(self, model: PreTrainedModel):
        return model.model.language_model.layers if hasattr(model, "model") else model.language_model.layers

    def get_model_layer(self, model: PreTrainedModel):
        return model.model.language_model if hasattr(model, "model") else model.language_model

    def prepare_forward_args(self, *args):
        args = list(args)
        input_ids = None if self.rbln_config.use_inputs_embeds else args.pop(0)
        inputs_embeds = args.pop(0) if self.rbln_config.use_inputs_embeds else None
        cache_position = args.pop(0)
        global_block_tables = args.pop(0)
        local_block_tables = None
        position_embeds = args.pop(0)
        query_position = args.pop(0) if self.phase == "prefill" and self.rbln_config.logits_to_keep > 0 else None
        position_ids = None
        attention_mask = args.pop(0) if self.rbln_config.use_attention_mask else None
        lora_int_id = args.pop(0) if self.rbln_config.lora_config else None
        past_key_values = args

        if len(past_key_values) != 2 * self.num_hidden_layers:
            raise ValueError(
                f"Different past_key_values to model's config. {len(past_key_values)} != {2 * self.num_hidden_layers}"
            )

        # [key, value] * n_layer -> ( (key, value) ) * n_layer
        # cache shape : batch, n_heads, 1, max_seq_len, head_dim
        _past_key_values = []
        for i in range(self.config.num_hidden_layers):
            key_states = past_key_values[i * 2]
            value_states = past_key_values[i * 2 + 1]
            past_key_value = [key_states, value_states]
            _past_key_values.append(past_key_value)
        past_key_values = _past_key_values

        return (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            query_position,
            attention_mask,
            position_ids,
            lora_int_id,
            past_key_values,
            position_embeds,
        )
