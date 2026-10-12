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

import torch

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
)


class GraniteWrapper(DecoderOnlyWrapper):
    def get_rbln_attn_class(self):
        return GraniteAttention

    def get_rbln_layer_class(self):
        return GraniteLayer

    def get_rbln_model_class(self):
        return GraniteModel

    def get_rbln_causal_lm_class(self):
        return GraniteForCausalLM


class GraniteAttention(DecoderOnlyAttention):
    def get_attn_scale(self, self_attn):
        return self_attn.scaling


class GraniteLayer(DecoderOnlyLayer):
    def __init__(self, layer, self_attn, lora_config=None):
        super().__init__(layer, self_attn, lora_config=lora_config)
        self.residual_multiplier = layer.residual_multiplier

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
        hidden_states = self.get_pre_attention_layernorm()(hidden_states)

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
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.get_post_attention_layernorm()(hidden_states)
        hidden_states = self.forward_mlp(hidden_states, lora_int_id)
        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states


class GraniteModel(DecoderOnlyModel):
    @property
    def hidden_multiplier(self):
        return self.config.embedding_multiplier


class GraniteForCausalLM(DecoderOnlyForCausalLM):
    def forward(self, *args, **kwargs):
        logits, all_hidden_states = super().forward(*args, **kwargs)
        return logits / self.config.logits_scaling, all_hidden_states
