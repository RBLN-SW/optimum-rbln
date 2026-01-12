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

from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import logging


logger = logging.get_logger(__name__)


class WhisperWrapper:
    def __init__(self, model, use_attention_mask, rbln_token_timestamps):
        self.encoder = WhisperEncoderWrapper(model)
        self.decoder = WhisperDecoderWrapper(
            model, use_attention_mask=use_attention_mask, output_attentions=rbln_token_timestamps
        )


class WhisperEncoderWrapper(torch.nn.Module):
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
        input_features: Optional[torch.LongTensor],
        b_idx: torch.Tensor,
        cross_key_values: torch.Tensor,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        # 1. get encoder last_hidden_states
        encoder_outputs = self.encoder(input_features=input_features)
        last_hidden_states = encoder_outputs[0]

        # 2. pre-compute cross_attention's past_key_value which used in decoder phase.
        cross_kv = []
        batch_size = input_features.shape[0]
        for k_proj, v_proj in zip(self.cross_k_projects, self.cross_v_projects):
            past_k = k_proj(last_hidden_states).view(batch_size, -1, self.num_heads, self.d_kv).transpose(1, 2)
            past_v = v_proj(last_hidden_states).view(batch_size, -1, self.num_heads, self.d_kv).transpose(1, 2)

            cross_kv.append(past_k)
            cross_kv.append(past_v)

        cross_kv = torch.stack(cross_kv, dim=0)

        # 3. update cross_attention's past_key_value to the device-dram for optimization.
        batch_axis = torch.tensor(1, dtype=torch.int16)
        cross_key_values = torch.ops.rbln_custom_ops.rbln_cache_update(
            cross_key_values, cross_kv, b_idx[0], batch_axis
        )

        return cross_key_values


class WhisperDecoderWrapper(torch.nn.Module):
    def __init__(self, model, use_attention_mask: bool = True, output_attentions: bool = False, **kwargs):
        super().__init__()
        self.config = model.config
        self.proj_out = model.proj_out
        self.use_attention_mask = use_attention_mask
        self.output_attentions = output_attentions
        self.__post_init__(model, **kwargs)

    def __post_init__(self, model: nn.Module, **kwargs):
        """
        Post-initialization to extract and configure encoder-related attributes.
        It is inspired by the BART architecture, but it is designed to be flexible and can be overridden
        by subclasses to modify or add custom attributes as necessary.
        """
        self.num_layers = self.config.decoder_layers
        self.decoder = self.convert_to_rbln_conditional_generation(model)

    def convert_to_rbln_conditional_generation(self, model: nn.Module):
        new_layers = []
        for layer in model.get_decoder().layers:
            self_attn = WhisperSelfAttention(layer.self_attn)
            cross_attn = WhisperCrossAttention(layer.encoder_attn)
            new_layers.append(WhisperDecoderLayer(layer, self_attn, cross_attn))

        decoder_model = WhisperDecoder(model.get_decoder(), new_layers)

        return decoder_model

    def forward(
        self,
        *args,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        if self.use_attention_mask:
            (
                decoder_input_ids,
                decoder_attention_mask,
                cache_position,
                block_tables,
                cross_kv_cache,
                *self_kv_cache,
            ) = args
        else:
            decoder_attention_mask = None
            (decoder_input_ids, cache_position, block_tables, cross_kv_cache, *self_kv_cache) = args

        # prepare past_key_values
        self_past_key_values = ()
        cross_past_key_values = ()
        for i in range(0, self.num_layers * 2, 2):
            self_past_key_values = self_past_key_values + ((self_kv_cache[i], self_kv_cache[i + 1]),)
            cross_past_key_values = cross_past_key_values + ((cross_kv_cache[i], cross_kv_cache[i + 1]),)

        # Decode
        sequence_output, cross_attentions = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            cache_position=cache_position,
            self_past_key_values=self_past_key_values,
            cross_past_key_values=cross_past_key_values,
            block_tables=block_tables,
        )

        lm_logits = self.proj_out(sequence_output)
        outputs = (lm_logits,)

        if self.output_attentions:
            # deocder's cross attention is used for token_timestamps
            cross_attention = torch.stack(cross_attentions, dim=0)
            outputs += (cross_attention,)

        return outputs


class WhisperDecoder(nn.Module):
    def __init__(self, model, layers, **kwargs):
        super().__init__()
        self._original_mod = model
        self.layers = nn.ModuleList(layers)
        self.embed_tokens = model.embed_tokens
        self.layer_norm = model.layer_norm
        self.embed_positions = model.embed_positions

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        self_past_key_values: Optional[torch.Tensor] = None,
        cross_past_key_values: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # positional embeding
        inputs_embeds = self.embed_tokens(input_ids)
        all_hiddens = []
        for i in range(inputs_embeds.shape[0]):
            position_id = cache_position[i]
            position = self.embed_positions.weight[position_id]
            batch_hidden = position + inputs_embeds[i]
            all_hiddens.append(batch_hidden)

        hidden_states = torch.cat(all_hiddens, dim=0).unsqueeze(1)

        # prepare attn mask (normal attention - masked)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]

        cross_attentions = ()
        # iterate decoder_layer
        for self_past_key_value, cross_past_key_value, decoder_layer in zip(
            self_past_key_values, cross_past_key_values, self.layers
        ):
            hidden_states, cross_attn_weights = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                self_past_key_value=self_past_key_value,
                cross_past_key_value=cross_past_key_value,
                cache_position=cache_position,
                block_tables=block_tables,
            )
            cross_attentions += (cross_attn_weights,)

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states, cross_attentions


class WhisperDecoderLayer(nn.Module):
    def __init__(self, decoder_layer, self_attn, cross_attn):
        super().__init__()
        self._original_mod = decoder_layer
        self.self_attn = self_attn
        self.encoder_attn = cross_attn
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
        block_tables: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self Attention Block
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_past_key_value,
            attention_mask=attention_mask,
            cache_position=cache_position,
            block_tables=block_tables,
        )
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            past_key_value=cross_past_key_value,
        )
        hidden_states = residual + hidden_states

        # Fully Connected Block
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, cross_attn_weights


class WhisperAttention(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self._original_mod = attn
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.out_proj = attn.out_proj
        self.num_heads = attn.num_heads
        self.embed_dim = attn.embed_dim
        self.head_dim = attn.embed_dim // attn.num_heads
        self.scaling = self.head_dim**-0.5

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


class WhisperSelfAttention(WhisperAttention):
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, 1, self.num_heads, self.head_dim).transpose(1, 3)

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

        args = {
            "q": query_states,
            "k": key_states,
            "v": value_states,
            "kcache": past_key_value[0].view(bsz, self.num_heads, 1, -1, self.head_dim),
            "vcache": past_key_value[1].view(bsz, self.num_heads, 1, -1, self.head_dim),
            "seq": cache_position.expand(bsz, 1),
            "scale": torch.tensor(1.0, dtype=torch.float32),
            "block_table": block_tables,
            "block_size": block_size,
        }

        if attention_mask is not None:
            args["mask"] = attention_mask.unsqueeze(2)
            attn_output = torch.ops.rbln_custom_ops.paged_attn_decode(**args)
        else:
            args["mask"] = None
            attn_output = torch.ops.rbln_custom_ops.paged_causal_attn_decode(**args)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class WhisperCrossAttention(WhisperAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, query_len, _ = hidden_states.size()
        query_states = self._shape(self.q_proj(hidden_states), query_len, batch_size)
        query_states = query_states * self.scaling

        key_states = past_key_value[0]
        value_states = past_key_value[1]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.view(batch_size, self.num_heads, query_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, query_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights
