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

import torch
from torch import nn

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyWrapper,
    RotaryEmbedding,
)
from ..decoderonly.lora_architecture import LoRALinear


class Llama4VisionWrapper(nn.Module):
    """Vision encoder with real-valued 2D RoPE for compilation."""

    def __init__(self, model):
        super().__init__()
        self.patch_embedding = model.patch_embedding
        self.class_embedding = model.class_embedding
        self.positional_embedding_vlm = model.positional_embedding_vlm
        self.layernorm_pre = model.layernorm_pre
        for layer in model.model.layers:
            layer.self_attn = Llama4VisionAttention(layer.self_attn)
        self.layers = model.model.layers
        self.layernorm_post = model.layernorm_post
        self.vision_adapter = model.vision_adapter
        frequencies = model.rotary_embedding.freqs_ci
        self.register_buffer("cos", frequencies.real.unsqueeze(0), persistent=False)
        self.register_buffer("sin", frequencies.imag.unsqueeze(0), persistent=False)
        head_dim = 2 * frequencies.shape[-1]
        pair_matrix = torch.zeros(head_dim, head_dim)
        pair_indices = torch.arange(head_dim // 2)
        pair_matrix[2 * pair_indices, 2 * pair_indices + 1] = 1
        pair_matrix[2 * pair_indices + 1, 2 * pair_indices] = -1
        self.register_buffer("pair_matrix", pair_matrix, persistent=False)

    def forward(self, pixel_values):
        hidden_states = self.patch_embedding(pixel_values)
        cls_embedding = self.class_embedding.expand(hidden_states.shape[0], 1, -1)
        hidden_states = torch.cat((hidden_states, cls_embedding), dim=1)
        hidden_states = self.layernorm_pre(hidden_states + self.positional_embedding_vlm)
        for layer in self.layers:
            hidden_states = layer(hidden_states, freqs_ci=(self.cos, self.sin, self.pair_matrix))[0]
        hidden_states = self.layernorm_post(hidden_states)[:, :-1]
        return self.vision_adapter(hidden_states)


class Llama4VisionAttention(nn.Module):
    def __init__(self, self_attn):
        super().__init__()
        self.num_heads = self_attn.num_heads
        self.head_dim = self_attn.head_dim
        self.q_proj = self_attn.q_proj
        self.k_proj = self_attn.k_proj
        self.v_proj = self_attn.v_proj
        self.o_proj = self_attn.o_proj

    def _rotate(self, states, cos, sin, pair_matrix):
        # Equivalent to HF's complex RoPE; 4D matmul keeps one device context.
        # The pair matrix maps each [real, imag] pair to [-imag, real].
        x = states.float()
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        return (x * cos + torch.matmul(x, pair_matrix.float()) * sin).to(states.dtype)

    def forward(self, hidden_states, freqs_ci, attention_mask=None):
        shape = (*hidden_states.shape[:-1], self.num_heads, self.head_dim)
        query = self._rotate(self.q_proj(hidden_states).view(shape), *freqs_ci).transpose(1, 2)
        key = self._rotate(self.k_proj(hidden_states).view(shape), *freqs_ci).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        weights = query @ key.transpose(-1, -2) * self.head_dim**-0.5
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = weights.softmax(dim=-1)
        attended = (weights @ value).transpose(1, 2).reshape(hidden_states.shape)
        return self.o_proj(attended), weights


class Llama4Wrapper(DecoderOnlyWrapper):
    def get_rotary_emb(self, max_seq_len):
        config = copy.deepcopy(self.config)
        config.rope_scaling = config.rope_parameters
        return RotaryEmbedding(config=config, max_seq_len_cached=max_seq_len)

    def get_rbln_attn_class(self):
        return Llama4Attention

    def get_rbln_layer_class(self):
        return Llama4DecoderLayer


class Llama4LoRALinear(LoRALinear):
    def get_layer_key(self) -> str:
        return f"base_model.model.model.layers.{self.layer_idx}.feed_forward.{self.projection_name}"


class Llama4DecoderLayer(DecoderOnlyLayer):
    _MLP_ATTR = ("feed_forward",)

    def get_lora_linear_class(self):
        return Llama4LoRALinear


class Llama4Attention(DecoderOnlyAttention):
    def __post_init__(self, self_attn):
        super().__post_init__(self_attn)
        self.use_rope = self_attn.use_rope
        self.qk_norm = getattr(self_attn, "qk_norm", None)
        self.attn_temperature_tuning = self_attn.attn_temperature_tuning
        self.floor_scale = self_attn.floor_scale
        self.attn_scale = self_attn.attn_scale
        self.attention_chunk_size = (
            self.config.attention_chunk_size
            if self.config.layer_types[self.layer_idx] == "chunked_attention"
            else None
        )

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        if not self.use_rope:
            return query_states, key_states

        # The shared lookup table stores split-half cos/sin values. Llama 4 uses
        # the same frequencies, but rotates adjacent pairs of head dimensions.
        cos = cos[..., : self.head_dim // 2].repeat_interleave(2, dim=-1)
        sin = sin[..., : self.head_dim // 2].repeat_interleave(2, dim=-1)

        def rotate_pairs(x):
            pairs = x.reshape(*x.shape[:-1], -1, 2)
            return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)

        query_dtype, key_dtype = query_states.dtype, key_states.dtype
        query_states = query_states.float()
        key_states = key_states.float()
        return (
            (query_states * cos + rotate_pairs(query_states) * sin).to(query_dtype),
            (key_states * cos + rotate_pairs(key_states) * sin).to(key_dtype),
        )

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
        batch_size, query_length, _ = hidden_states.size()
        query_states, key_states, value_states = self.projection(hidden_states, lora_int_id)
        query_states = query_states.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, query_length, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )

        if batch_size > 1 and "prefill" in self.phase:
            raise NotImplementedError(f"batch size should be 1 if prefill phase, but got {batch_size}.")

        if cos is not None and sin is not None:
            query_states, key_states = self.apply_rotary_pos_embed(query_states, key_states, cos, sin)
        if self.qk_norm is not None:
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        positions = seq_positions[:, :1] + torch.arange(query_length, device=hidden_states.device)
        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = torch.log1p(torch.floor((positions.float() + 1.0) / self.floor_scale))
            attn_scales = (attn_scales * self.attn_scale + 1.0)[:, None, :, None]
            query_states = (query_states * attn_scales).to(query_states.dtype)

        if self.attention_chunk_size is not None and self.rbln_config.max_seq_len > self.attention_chunk_size:
            key_positions = torch.arange(self.rbln_config.max_seq_len, device=hidden_states.device)
            chunk_mask = (
                positions[:, :, None] // self.attention_chunk_size == key_positions // self.attention_chunk_size
            )
            chunk_mask = chunk_mask & (key_positions <= positions[:, :, None])
            attention_mask = attention_mask * chunk_mask[:, None, :, :].to(attention_mask.dtype)

        k_scale, v_scale = self.maybe_get_kvcache_scale()
        attn_output = self.get_attention_op()(
            query_states,
            key_states,
            value_states,
            attention_mask,
            past_key_state=past_key_values[self.layer_idx][0],
            past_value_state=past_key_values[self.layer_idx][1],
            seq_position=seq_positions,
            scale=self.scale,
            block_tables=block_tables,
            block_size=self.kvcache_block_size,
            k_scale=k_scale,
            v_scale=v_scale,
            s_aux=getattr(self, "sinks", None),
        )
        return self.o_proj(attn_output, lora_int_id) if self.lora_config else self.o_proj(attn_output)
