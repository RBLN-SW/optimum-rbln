# Copyright 2026 Rebellions Inc. All rights reserved.
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

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyWrapper,
    apply_rotary_pos_emb,
)
from .configuration_exaone4_5 import RBLNExaone4_5_VisionModelConfig


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class Exaone4_5VisionTransformerWrapper(nn.Module):
    def __init__(self, model: torch.nn.Module, rbln_config: RBLNExaone4_5_VisionModelConfig):
        super().__init__()
        self.fullatt_block_indexes = model.fullatt_block_indexes
        self.merger = model.merger
        self.rbln_config = rbln_config
        window_seq_len = (model.window_size // model.patch_size) ** 2
        self.blocks = self.wrap_vision_blocks(model.blocks, window_seq_len, rbln_config)

    def wrap_vision_blocks(
        self,
        blocks: torch.nn.ModuleList,
        window_seq_len: int,
        rbln_config: RBLNExaone4_5_VisionModelConfig,
    ):
        wrapped_blocks = []
        for i, block in enumerate(blocks):
            is_full_attn = i in self.fullatt_block_indexes
            wrapped_blocks.append(Exaone4_5VisionBlock(block, is_full_attn, window_seq_len, rbln_config))
        return nn.ModuleList(wrapped_blocks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        full_attn_masks: torch.Tensor,
        window_attn_masks: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        full_attn_masks = full_attn_masks > 0
        window_attn_masks = window_attn_masks > 0

        for i, block in enumerate(self.blocks):
            attn_masks = full_attn_masks if i in self.fullatt_block_indexes else window_attn_masks
            hidden_states = block(hidden_states, attn_masks, [cos, sin])

        hidden_states = self.merger(hidden_states)
        return hidden_states


class Exaone4_5VisionBlock(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        is_full_attn: bool,
        window_seq_len: int,
        rbln_config: RBLNExaone4_5_VisionModelConfig,
    ):
        super().__init__()
        self._origin_model = model
        self.rbln_config = rbln_config
        self.norm1 = model.norm1
        self.norm2 = model.norm2

        if is_full_attn:
            self.attn = Exaone4_5VisionFullAttention(model.attn, rbln_config)
        else:
            self.attn = Exaone4_5VisionWindowAttention(model.attn, window_seq_len, rbln_config)
        self.mlp = model.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_masks: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attn_masks,
            position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Exaone4_5VisionFullAttention(nn.Module):
    def __init__(self, model: nn.Module, rbln_config: RBLNExaone4_5_VisionModelConfig) -> None:
        super().__init__()
        self._origin_model = model
        self.rbln_config = rbln_config
        self.num_heads = model.num_heads
        self.num_key_value_heads = getattr(model, "num_key_value_heads", model.num_heads)
        self.head_dim = getattr(model, "head_dim", model.proj.in_features // model.num_heads)
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.qkv = model.qkv
        self.proj = model.proj
        self.scale = torch.tensor(1 / math.sqrt(self.head_dim), dtype=rbln_config.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_masks: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        if self.num_key_value_heads == self.num_heads:
            hidden_states = hidden_states.unsqueeze(0)
            q, k, v = (
                self.qkv(hidden_states).reshape(1, seq_length, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
            )
        else:
            qkv = self.qkv(hidden_states)
            q, k, v = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

            q = q.view(seq_length, self.num_heads, self.head_dim).permute(1, 0, 2).unsqueeze(0)
            k = k.view(seq_length, self.num_key_value_heads, self.head_dim)
            v = v.view(seq_length, self.num_key_value_heads, self.head_dim)
            repeat_factor = self.num_heads // self.num_key_value_heads
            k = k.repeat_interleave(repeat_factor, dim=1).permute(1, 0, 2).unsqueeze(0)
            v = v.repeat_interleave(repeat_factor, dim=1).permute(1, 0, 2).unsqueeze(0)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_masks,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale.item(),
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(1, seq_length, -1)
        attn_output = self.proj(attn_output).squeeze(0)
        return attn_output


class Exaone4_5VisionWindowAttention(nn.Module):
    def __init__(self, model: nn.Module, window_seq_len: int, rbln_config: RBLNExaone4_5_VisionModelConfig) -> None:
        super().__init__()
        self._origin_model = model
        self.rbln_config = rbln_config
        self.num_heads = model.num_heads
        self.num_key_value_heads = getattr(model, "num_key_value_heads", model.num_heads)
        self.head_dim = getattr(model, "head_dim", model.proj.in_features // model.num_heads)
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.qkv = model.qkv
        self.proj = model.proj
        self.window_seq_len = window_seq_len
        self.scale = torch.tensor(1 / math.sqrt(self.head_dim), dtype=rbln_config.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_masks: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length, hidden_dim = hidden_states.shape
        assert seq_length % self.window_seq_len == 0
        num_windows = seq_length // self.window_seq_len
        hidden_states = hidden_states.reshape(num_windows, self.window_seq_len, hidden_dim)

        if self.num_key_value_heads == self.num_heads:
            q, k, v = (
                self.qkv(hidden_states)
                .reshape(num_windows, self.window_seq_len, 3, self.num_heads, -1)
                .permute(2, 0, 3, 1, 4)
                .unbind(0)
            )
        else:
            qkv = self.qkv(hidden_states)
            q, k, v = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
            q = q.view(num_windows, self.window_seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(num_windows, self.window_seq_len, self.num_key_value_heads, self.head_dim)
            v = v.view(num_windows, self.window_seq_len, self.num_key_value_heads, self.head_dim)
            repeat_factor = self.num_heads // self.num_key_value_heads
            k = k.repeat_interleave(repeat_factor, dim=2).permute(0, 2, 1, 3)
            v = v.repeat_interleave(repeat_factor, dim=2).permute(0, 2, 1, 3)

        cos, sin = position_embeddings
        cos = cos.reshape(num_windows, 1, seq_length // num_windows, -1)
        sin = sin.reshape(num_windows, 1, seq_length // num_windows, -1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_masks,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale.item(),
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(1, seq_length, -1)
        attn_output = self.proj(attn_output).squeeze(0)
        return attn_output


class Exaone4_5LanguageModelWrapper(DecoderOnlyWrapper):
    def get_decoder_layers(self, model: "PreTrainedModel"):
        return model.get_decoder().layers

    def get_model_layer(self, model: "PreTrainedModel"):
        return model.get_decoder()

    def get_rbln_attn_class(self):
        return Exaone4_5Attention

    def get_rbln_layer_class(self):
        return Exaone4_5DecoderLayer


class Exaone4_5Attention(DecoderOnlyAttention):
    def __post_init__(self, self_attn):
        self.q_proj = self_attn.q_proj
        self.k_proj = self_attn.k_proj
        self.v_proj = self_attn.v_proj
        self.o_proj = self_attn.o_proj
        self.q_norm = self_attn.q_norm
        self.k_norm = self_attn.k_norm

    def get_attn_scale(self, self_attn):
        return getattr(self_attn, "scaling", self_attn.head_dim**-0.5)

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        if self.is_sliding:
            return super().apply_rotary_pos_embed(query_states, key_states, cos, sin)
        return query_states, key_states


class Exaone4_5DecoderLayer(DecoderOnlyLayer):
    _PRE_ATTN_LAYERNORM = None
    _POST_ATTN_LAYERNORM = ["post_attention_layernorm"]
    _PRE_FF_LAYERNORM_ATTRS = None
    _POST_FF_LAYERNORM_ATTRS = ["post_feedforward_layernorm"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_positions: torch.LongTensor,
        past_key_values: Tuple[Tuple[torch.Tensor]],
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        lora_int_id: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            seq_positions=seq_positions,
            past_key_values=past_key_values,
            cos=cos,
            sin=sin,
            block_tables=block_tables,
            lora_int_id=lora_int_id,
        )
        hidden_states = self.get_post_attention_layernorm()(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.forward_mlp(hidden_states, lora_int_id)
        hidden_states = self.get_post_feedforward_layernorm()(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
