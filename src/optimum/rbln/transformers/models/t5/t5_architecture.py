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

from typing import Tuple

import torch
from torch import nn
from transformers.utils import logging

from ..seq2seq.seq2seq_architecture import (
    Seq2SeqDecoder,
    Seq2SeqDecoderLayer,
    Seq2SeqDecoderWrapper,
    Seq2SeqEncoderWrapper,
    Seq2SeqForConditionalGeneration,
    Seq2SeqSelfAttention,
)


logger = logging.get_logger(__name__)


class T5Wrapper:
    def __init__(self, model: nn.Module, enc_max_seq_len: int, dec_max_seq_len: int = None):
        self.encoder = T5EncoderWrapper(model, enc_max_seq_len)
        self.decoder = T5DecoderWrapper(model, dec_max_seq_len=dec_max_seq_len)


class T5EncoderWrapper(Seq2SeqEncoderWrapper):
    def __post_init__(self, model: nn.Module):
        self.n_layer = self.config.num_layers
        self.cross_k_projects, self.cross_v_projects = self._extract_cross_kv_projects(model.get_decoder().block)
        self.num_heads = self.config.num_heads
        self.d_kv = self.config.d_kv

    def _extract_cross_kv_projects(self, t5_block: nn.Module):
        return (
            # different from bart
            nn.ModuleList(t5_block[i].layer[1].EncDecAttention.k for i in range(self.n_layer)),
            nn.ModuleList(t5_block[i].layer[1].EncDecAttention.v for i in range(self.n_layer)),
        )


class T5DecoderWrapper(Seq2SeqDecoderWrapper):
    def __post_init__(self, model, dec_max_seq_len: int = None):
        self.num_layers = self.config.num_layers
        self.conditional_generation = self.convert_to_rbln_conditional_generation(model, dec_max_seq_len)

    def convert_to_rbln_conditional_generation(self, model: nn.Module, dec_max_seq_len: int):
        new_blocks = []
        for block in model.get_decoder().block:
            self_attn = T5LayerSelfAttention(block.layer[0].SelfAttention)
            block = T5Block(block, self_attn)
            new_blocks.append(block)

        decoder_model = T5Decoder(model.get_decoder(), new_blocks, dec_max_seq_len=dec_max_seq_len)
        new_model = T5ForConditionalGeneration(model, decoder_model)

        return new_model

    def forward(
        self,
        input_ids,
        attention_mask,
        encoder_attention_mask,
        cache_position,
        block_tables,
        *kv_cache,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor]]:
        self_past_key_values = ()
        cross_past_key_values = ()
        self_kv_cache = kv_cache[self.num_layers * 2 :]
        cross_kv_cache = kv_cache[: self.num_layers * 2]

        for i in range(0, self.num_layers * 2, 2):
            self_past_key_values = self_past_key_values + ((self_kv_cache[i], self_kv_cache[i + 1]),)
            cross_past_key_values = cross_past_key_values + ((cross_kv_cache[i], cross_kv_cache[i + 1]),)

        # decode
        lm_logits = self.conditional_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            self_past_key_values=self_past_key_values,
            cross_past_key_values=cross_past_key_values,
            cache_position=cache_position,
            block_tables=block_tables,
        )

        return lm_logits


class T5ForConditionalGeneration(Seq2SeqForConditionalGeneration):
    has_rescaling = True

    def __post_init__(self):
        self.scaling = self.config.d_model**-0.5


class T5Decoder(Seq2SeqDecoder):
    has_pos_emb = False

    def __post_init__(self, dec_max_seq_len: int = None):
        self.invert_attention_mask = self._original_mod.invert_attention_mask
        self._dec_position_bias = self.precompute_dec_position_bias(self._original_mod, dec_max_seq_len)

    def precompute_dec_position_bias(self, model, dec_max_length):
        attn_layer = model.block[0].layer[0].SelfAttention
        return attn_layer.compute_bias(dec_max_length, dec_max_length)

    def prepare_attn_mask(self, attention_mask, encoder_attention_mask, cache_position):
        attention_mask = self.invert_attention_mask(attention_mask)
        encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        b_size = attention_mask.shape[0]
        batch_decoder_position_bias = []
        for i in range(b_size):
            if torch.compiler.is_exporting():
                cache_pos = cache_position[i][0].item()
                torch._check_is_size(cache_pos)
                torch._check(cache_pos >= 0)
                torch._check(cache_pos < self._dec_position_bias.shape[2])
            else:
                cache_pos = cache_position[i][0]
            batch_position_bias = torch.select(self._dec_position_bias, dim=2, index=cache_pos).unsqueeze(2)
            batch_decoder_position_bias.append(batch_position_bias)
        position_bias = torch.cat(batch_decoder_position_bias, dim=0)

        attention_mask = position_bias + attention_mask

        return attention_mask, encoder_attention_mask


class T5Block(Seq2SeqDecoderLayer):
    def __init__(self, decoder_layer, self_attn):
        super().__init__(decoder_layer, self_attn, cross_attn=None)
        self.__post_init__()

    def __post_init__(self):
        self.self_attn_layer_norm = self._original_mod.layer[0].layer_norm
        self.encoder_attn_layer_norm = self._original_mod.layer[1].layer_norm
        self.cross_attn = T5CrossAttention(self._original_mod.layer[1].EncDecAttention)
        self.ff_layer = self._original_mod.layer[2]

    def pre_self_attn_layer_norm(self, hidden_states):
        return self.self_attn_layer_norm(hidden_states)

    def post_self_attn_layer_norm(self, hidden_states):
        return hidden_states

    def pre_cross_attn_layer_norm(self, hidden_states):
        return self.encoder_attn_layer_norm(hidden_states)

    def post_cross_attn_layer_norm(self, hidden_states):
        return hidden_states


class T5LayerSelfAttention(Seq2SeqSelfAttention):
    def __post_init__(self):
        self.q_proj = self._original_mod.q
        self.k_proj = self._original_mod.k
        self.v_proj = self._original_mod.v
        self.out_proj = self._original_mod.o
        self.num_heads = self._original_mod.n_heads
        self.head_dim = self._original_mod.key_value_proj_dim
        self.attn_decode = torch.ops.rbln_custom_ops.paged_add_softmax_attn_decode

    def projection(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Tuple[torch.Tensor],
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        block_tables: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        bsz, tgt_len, _ = hidden_states.size()

        query_states, key_states, value_states = self.projection(hidden_states=hidden_states)
        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        block_size = past_key_value[0].shape[-2]
        attn_output = self.attn_decode(
            query_states,
            key_states,
            value_states,
            attention_mask.unsqueeze(
                2
            ),  # Unsqueeze group axis since CustomKernel expects it for group query attention
            past_key_value[0].view(bsz, self.num_heads, 1, -1, self.head_dim),
            past_key_value[1].view(bsz, self.num_heads, 1, -1, self.head_dim),
            cache_position,
            torch.tensor(1.0, dtype=torch.float32),  # scale
            block_tables,
            block_size,
        )

        attn_output = attn_output.view(bsz, self.num_heads, -1, self.head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(bsz, -1, self.num_heads * self.head_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output


class T5CrossAttention(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.q = attn.q
        self.o = attn.o
        self.n_heads = attn.n_heads
        self.key_value_proj_dim = attn.key_value_proj_dim
        self.inner_dim = attn.inner_dim

    def forward(
        self,
        hidden_states: torch.Tensor = None,
        past_key_value: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        key_value_states: torch.Tensor = None,
    ):
        batch_size = hidden_states.shape[0]

        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        scores += attention_mask

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output, past_key_value)

        return outputs
