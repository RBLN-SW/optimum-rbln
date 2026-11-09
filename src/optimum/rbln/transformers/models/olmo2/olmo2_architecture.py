# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyWrapper,
    NestedCompileRegionMixin,
)


class Olmo2Wrapper(DecoderOnlyWrapper):
    def get_rbln_attn_class(self):
        return Olmo2Attention

    def get_rbln_layer_class(self):
        return Olmo2DecoderLayerRegion if self.rbln_config.nested_compile_region else Olmo2DecoderLayer


class Olmo2Attention(DecoderOnlyAttention):
    def __post_init__(self, self_attn):
        super().__post_init__(self_attn)
        # OLMo-2 normalizes q/k over all heads, so the norm has to run here, before the head reshape.
        self.full_q_norm = self_attn.q_norm
        self.full_k_norm = self_attn.k_norm

    def projection(
        self, hidden_states, lora_int_id: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states, key_states, value_states = super().projection(hidden_states, lora_int_id)
        if query_states.shape != key_states.shape:
            return self.full_q_norm(query_states), self.full_k_norm(key_states), value_states

        # One reduction over a stacked q/k pair instead of one per norm: each full-projection norm is an
        # all-reduce of its statistic once the heads are sharded.
        dtype = query_states.dtype
        stacked = torch.stack([query_states, key_states]).float()
        stacked = stacked * torch.rsqrt(stacked.pow(2).mean(-1, keepdim=True) + self.full_q_norm.variance_epsilon)
        query_states = (self.full_q_norm.weight * stacked[0]).to(dtype)
        key_states = (self.full_k_norm.weight * stacked[1]).to(dtype)
        return query_states, key_states, value_states


class Olmo2DecoderLayer(DecoderOnlyLayer):
    _PRE_ATTN_LAYERNORM = None
    _POST_ATTN_LAYERNORM = ["post_attention_layernorm"]
    _POST_FF_LAYERNORM_ATTRS = ["post_feedforward_layernorm"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_positions: torch.LongTensor,
        past_key_values: tuple[tuple[torch.Tensor]],
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        block_tables: torch.Tensor | None = None,
        lora_int_id: torch.Tensor | None = None,
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

        residual = hidden_states
        hidden_states = self.forward_mlp(hidden_states, lora_int_id)
        hidden_states = self.get_post_feedforward_layernorm()(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Olmo2DecoderLayerRegion(NestedCompileRegionMixin, Olmo2DecoderLayer):
    """Decoder layer with a nested compile region around its forward call."""
