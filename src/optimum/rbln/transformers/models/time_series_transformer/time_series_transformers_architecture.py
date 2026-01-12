# Copyright 2024 Rebellions Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this software are licensed under the Apache License,
# Version 2.0. See the NOTICE file distributed with this work for
# additional information regarding copyright ownership.

# All other portions of this software, including proprietary code,
# are the intellectual property of Rebellions Inc. and may not be
# copied, modified, or distributed without prior written permission
# from Rebellions Inc.

from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import logging


logger = logging.get_logger(__name__)


class TimeSeriesTransformersWrapper:
    def __init__(self, model, num_parallel_samples):
        self.encoder = TimeSeriesTransformersEncoderWrapper(model)
        self.decoder = TimeSeriesTransformersDecoderWrapper(model, num_parallel_samples)


class TimeSeriesTransformersEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.config = model.config
        self.encoder = model.get_encoder()
        self.num_heads = self.config.decoder_attention_heads
        self.d_kv = self.config.d_model // self.num_heads
        self.cross_k_projects, self.cross_v_projects = self._extract_cross_kv_projects(model.get_decoder().layers)

    def _extract_cross_kv_projects(self, decoder_layers: nn.Module):
        return (
            nn.ModuleList(layer.encoder_attn.k_proj for layer in decoder_layers),
            nn.ModuleList(layer.encoder_attn.v_proj for layer in decoder_layers),
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cross_key_values: torch.Tensor,  # n_layers, batch_size, num_heads, context_length, d_kv
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        # 1. get encoder last_hidden_states
        encoder_outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=None, return_dict=False)
        last_hidden_states = encoder_outputs[0]

        # 2. pre-compute cross_attention's past_key_value which used in decoder phase.
        cross_kv = []
        batch_size = inputs_embeds.shape[0]
        for k_proj, v_proj in zip(self.cross_k_projects, self.cross_v_projects):
            past_k = k_proj(last_hidden_states).view(batch_size, -1, self.num_heads, self.d_kv).transpose(1, 2)
            past_v = v_proj(last_hidden_states).view(batch_size, -1, self.num_heads, self.d_kv).transpose(1, 2)

            cross_kv.append(past_k)
            cross_kv.append(past_v)

        cross_kv = torch.stack(cross_kv, dim=0)

        # 3. update cross_attention's past_key_value to the device-dram for optimization.
        bidx = torch.tensor(0, dtype=torch.int16)
        axis = torch.tensor(1, dtype=torch.int16)
        enc_output = torch.ops.rbln_custom_ops.rbln_cache_update(cross_key_values, cross_kv, bidx, axis)

        return enc_output


class TimeSeriesTransformersDecoderWrapper(torch.nn.Module):
    def __init__(self, model, num_parallel_samples):
        super().__init__()
        self.config = model.config
        self.num_layers = self.config.decoder_layers
        self.decoder = self.convert_to_rbln_tst_decoder(model, num_parallel_samples)
        self.parameter_projection = model.parameter_projection

    def convert_to_rbln_tst_decoder(self, model: nn.Module, num_parallel_samples: int):
        new_layers = []
        for layer in model.get_decoder().layers:
            self_attn = TimeSeriesTransformersSelfAttention(layer.self_attn, num_parallel_samples)
            cross_attn = TimeSeriesTransformersCrossAttention(layer.encoder_attn, num_parallel_samples)
            new_layers.append(TimeSeriesTransformersDecoderLayer(layer, self_attn, cross_attn))

        decoder_model = TimeSeriesTransformersDecoder(model.get_decoder(), new_layers)

        return decoder_model

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        block_tables: torch.Tensor,
        cross_kv_cache: torch.Tensor,  # batch_size, num_heads, context_length, d_kv
        *self_kv_cache: torch.Tensor,  # batch_size * num_parallel_samples, num_heads, prediction_length, d_kv
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        # prepare past_key_values
        self_past_key_values = ()
        cross_past_key_values = ()
        for i in range(0, self.num_layers * 2, 2):
            self_past_key_values = self_past_key_values + ((self_kv_cache[i], self_kv_cache[i + 1]),)
            cross_past_key_values = cross_past_key_values + ((cross_kv_cache[i], cross_kv_cache[i + 1]),)

        # Decode
        last_hidden_states = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=decoder_attention_mask,
            cache_position=cache_position,
            block_tables=block_tables,
            self_past_key_values=self_past_key_values,
            cross_past_key_values=cross_past_key_values,
        )

        params = self.parameter_projection(last_hidden_states[:, -1:])

        outputs = ()
        outputs += (params,)
        outputs += (last_hidden_states,)

        return outputs


class TimeSeriesTransformersDecoder(nn.Module):
    def __init__(self, model, layers, **kwargs):
        super().__init__()
        self._original_mod = model
        self.config = model.config
        self.layers = nn.ModuleList(layers)
        self.value_embedding = model.value_embedding
        self.embed_positions = model.embed_positions
        self.layernorm_embedding = model.layernorm_embedding

    def forward(
        self,
        inputs_embeds: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        self_past_key_values: Optional[torch.Tensor] = None,
        cross_past_key_values: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        block_tables: torch.Tensor = None,
    ):
        input_shape = inputs_embeds.size()[:-1]

        # prepare casual_attn_mask
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, cache_position)

        hidden_states = self.value_embedding(inputs_embeds)
        embed_idx = cache_position + self.config.context_length
        if torch.compiler.is_exporting():
            embed_idx = embed_idx.item()
            torch._check_is_size(embed_idx)
            torch._check(embed_idx >= 0)
            torch._check(embed_idx < len(self.embed_positions.weight))
        embed_pos = self.embed_positions.weight[embed_idx]
        hidden_states = self.layernorm_embedding(hidden_states + embed_pos)

        # iterate decoder_layer
        for self_past_key_value, cross_past_key_value, decoder_layer in zip(
            self_past_key_values, cross_past_key_values, self.layers
        ):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                self_past_key_value=self_past_key_value,
                cross_past_key_value=cross_past_key_value,
                cache_position=cache_position,
                block_tables=block_tables,
            )

        return hidden_states


class TimeSeriesTransformersDecoderLayer(nn.Module):
    def __init__(self, decoder_layer, self_attn, cross_attn):
        super().__init__()
        self._original_mod = decoder_layer
        self.self_attn = self_attn
        self.encoder_attn = cross_attn
        self.embed_dim = decoder_layer.embed_dim
        self.self_attn_layer_norm = decoder_layer.self_attn_layer_norm
        self.encoder_attn_layer_norm = decoder_layer.encoder_attn_layer_norm
        self.final_layer_norm = decoder_layer.final_layer_norm
        self.activation_fn = decoder_layer.activation_fn
        self.fc1 = decoder_layer.fc1
        self.fc2 = decoder_layer.fc2

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        self_past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cross_past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,
        block_tables: torch.Tensor = None,
    ) -> torch.Tensor:
        # Self Attention Block
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_past_key_value,
            attention_mask=attention_mask,
            cache_position=cache_position,
            block_tables=block_tables,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        residual = hidden_states
        hidden_states = self.encoder_attn(
            hidden_states=hidden_states,
            past_key_value=cross_past_key_value,
            # attention_mask=encoder_attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected Block
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class TimeSeriesTransformersAttention(nn.Module):
    def __init__(self, attn, num_parallel_samples):
        super().__init__()
        self._original_mod = attn
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.out_proj = attn.out_proj
        self.num_heads = attn.num_heads
        self.embed_dim = attn.embed_dim
        self.head_dim = attn.head_dim
        self.scaling = attn.scaling
        self.num_parallel_samples = num_parallel_samples

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


class TimeSeriesTransformersSelfAttention(TimeSeriesTransformersAttention):
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(1, seq_len, 1, bsz * self.num_heads, self.head_dim).transpose(1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)
        query_states = query_states * self.scaling

        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        block_size = past_key_value[0].shape[-2]
        attn_output = torch.ops.rbln_custom_ops.paged_add_softmax_attn_decode(
            q=query_states,
            k=key_states,
            v=value_states,
            mask=attention_mask.unsqueeze(2),
            kcache=past_key_value[0].view(1, bsz * self.num_heads, 1, -1, self.head_dim),
            vcache=past_key_value[1].view(1, bsz * self.num_heads, 1, -1, self.head_dim),
            seq=cache_position.expand(bsz, 1),
            scale=torch.tensor(1.0, dtype=torch.float32),  # scale
            block_table=block_tables,
            block_size=block_size,
        )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class TimeSeriesTransformersCrossAttention(TimeSeriesTransformersSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, query_len, _ = hidden_states.size()
        query_states = (
            self.q_proj(hidden_states)
            .view(
                batch_size // self.num_parallel_samples,
                self.num_parallel_samples,
                query_len,
                self.num_heads,
                self.head_dim,
            )
            .transpose(2, 3)
        )
        query_states = query_states * self.scaling

        key_states = past_key_value[0].unsqueeze(1)
        value_states = past_key_value[1].unsqueeze(1)

        attn_weights = torch.matmul(query_states, key_states.transpose(3, 4))
        attn_weights = attn_weights
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.view(batch_size, self.num_heads, query_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, query_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output
