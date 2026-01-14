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

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from transformers import PhiForCausalLM

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyWrapper,
    apply_rotary_pos_emb_partial,
)


if TYPE_CHECKING:
    from transformers import PhiForCausalLM, PhiModel


class PhiWrapper(DecoderOnlyWrapper):
    def get_rbln_attn_class(self):
        return PhiAttention

    def get_rbln_layer_class(self):
        return PhiLayer

    def get_model_layer(self, model: Union["PhiForCausalLM", "PhiModel"]):
        return model.model if self.is_causal_lm else model

    def get_decoder_layers(self, model: Union["PhiForCausalLM", "PhiModel"]):
        return model.model.layers if self.is_causal_lm else model.layers


class PhiAttention(DecoderOnlyAttention):
    _Q_NORM_ATTRS = ["q_layernorm"]
    _K_NORM_ATTRS = ["k_layernorm"]

    def __post_init__(self, self_attn):
        self.rotary_ndims = self_attn.rotary_ndims

    def projection(self, hidden_states, lora_int_id) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if lora_int_id is not None:
            raise NotImplementedError("LoRA is not supported for PhiAttention")

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if getattr(self, "q_norm", None) is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        return query_states, key_states, value_states

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        return apply_rotary_pos_emb_partial(query_states, key_states, cos, sin, ndim=self.rotary_ndims)


class PhiLayer(DecoderOnlyLayer):
    _POST_ATTN_LAYERNORM = None

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

        hidden_states = self.get_pre_attention_layernorm()(hidden_states)

        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            seq_positions=seq_positions,
            past_key_values=past_key_values,
            cos=cos,
            sin=sin,
            block_tables=block_tables,
        )

        feed_forward_hidden_states = self.mlp(hidden_states)

        hidden_states = attn_output + feed_forward_hidden_states + residual

        return hidden_states
