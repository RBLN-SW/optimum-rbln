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

import copy
from typing import Optional, Tuple, Union

import torch
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
    RotaryEmbedding,
    slice_and_unsqueeze_cos_sin,
)


class Gemma3ForCausalLMWrapper(DecoderOnlyWrapper):
    def get_rotary_emb(self, max_seq_len):
        rotary_emb_global = RotaryEmbedding(config=self.config, max_seq_len_cached=max_seq_len)

        config = copy.deepcopy(self.config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        rotary_emb_local = RotaryEmbedding(config=config, max_seq_len_cached=max_seq_len)

        return (rotary_emb_global, rotary_emb_local)

    def get_rbln_attn_class(self):
        return Gemma3Attention

    def get_rbln_layer_class(self):
        return Gemma3DecoderLayer

    def get_rbln_model_class(self):
        return Gemma3TextModel


class Gemma3TextModel(DecoderOnlyModel):
    # Different from DecoderOnlyModel, this model has global and local rotary embeddings.
    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        query_position: torch.Tensor = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        rotary_emb: torch.nn.Module = None,
        global_block_tables: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
        lora_int_id: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        # retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        # embed positions
        if inputs_embeds is None:
            inputs_embeds = self.get_embedding()(input_ids)

        hidden_states = inputs_embeds

        # Global Position Embeddings
        cos_global, sin_global = rotary_emb[0](hidden_states, self.max_seq_len)
        cos_global, sin_global = slice_and_unsqueeze_cos_sin(cos_global, sin_global, position_ids)

        # Local Position Embeddings
        cos_local, sin_local = rotary_emb[1](hidden_states, self.max_seq_len)
        cos_local, sin_local = slice_and_unsqueeze_cos_sin(cos_local, sin_local, position_ids)

        # (batch, seq_len) -> (batch,)
        if self.attn_impl == "flash_attn":
            seq_positions = cache_position[:, 0]
            seq_positions = self.convert_sequence_positions_for_flash_attn(
                seq_positions=seq_positions, max_seq_len=self.max_seq_len
            )
        else:
            seq_positions = cache_position[:, :1]

        cache_seq_len, cache_offset, swa_attn_mask = self.get_swa_custom_op_args(position_ids, query_position)
        sliding_cache_pos = (cache_seq_len, cache_offset)

        all_hidden_states = () if output_hidden_states else None
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            is_sliding = True if layer_idx in self.sliding_window_layers else False
            is_sliding_decode = is_sliding and self.phase == "decode"
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=swa_attn_mask if is_sliding_decode else attention_mask,
                seq_positions=sliding_cache_pos if is_sliding else seq_positions,
                past_key_values=past_key_values,
                cos=cos_local if is_sliding else cos_global,
                sin=sin_local if is_sliding else sin_global,
                block_tables=local_block_tables if is_sliding else global_block_tables,
                lora_int_id=lora_int_id,
            )

        hidden_states = self.get_last_layernorm()(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return hidden_states, all_hidden_states


class Gemma3DecoderLayer(DecoderOnlyLayer):
    def get_pre_feedforward_layernorm(self) -> Gemma3RMSNorm:
        return self._original_mod.pre_feedforward_layernorm

    def get_post_feedforward_layernorm(self) -> Gemma3RMSNorm:
        return self._original_mod.post_feedforward_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_positions: Union[torch.LongTensor, Tuple[torch.LongTensor]],
        past_key_values: Tuple[Tuple[torch.Tensor]],
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        lora_int_id: Optional[torch.Tensor] = None,
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
        hidden_states = self.get_post_attention_layernorm()(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.get_pre_feedforward_layernorm()(hidden_states)
        hidden_states = self.forward_mlp(hidden_states, lora_int_id)
        hidden_states = self.get_post_feedforward_layernorm()(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3Attention(DecoderOnlyAttention):
    def __post_init__(self):
        self.q_proj = self._original_mod.q_proj
        self.k_proj = self._original_mod.k_proj
        self.v_proj = self._original_mod.v_proj
        self.o_proj = self._original_mod.o_proj
        self.q_norm = self._original_mod.q_norm
        self.k_norm = self._original_mod.k_norm

    def get_attn_scale(self):
        return self._original_mod.config.query_pre_attn_scalar**-0.5
