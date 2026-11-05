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
from torch import nn

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
    NestedCompileRegionMixin,
    RotaryEmbedding,
    slice_and_unsqueeze_cos_sin,
)


def uses_chunked_attention(text_config) -> bool:
    # attention_chunk_size=None (e.g. Llama Guard 4) means every layer is full attention,
    # whatever HF's layer_types default says.
    return text_config.attention_chunk_size is not None and "chunked_attention" in text_config.layer_types


class Llama4VisionFullAttention(nn.Module):
    def __init__(self, self_attn):
        super().__init__()
        self.num_heads = self_attn.num_heads
        self.head_dim = self_attn.head_dim
        self.q_proj = self_attn.q_proj
        self.k_proj = self_attn.k_proj
        self.v_proj = self_attn.v_proj
        self.o_proj = self_attn.o_proj

    def _rotate(self, states, cos, sin, pair_matrix):
        # Real-valued form of HF's complex RoPE; low-precision results can differ slightly.
        # The pair matrix maps each [real, imag] pair to [-imag, real].
        x = states.float()
        cos, sin = cos.float(), sin.float()
        return (x * cos + torch.matmul(x, pair_matrix.float()) * sin).to(states.dtype)

    def forward(self, hidden_states, cos, sin, pair_matrix):
        shape = (*hidden_states.shape[:-1], self.num_heads, self.head_dim)
        query = self._rotate(self.q_proj(hidden_states).view(shape), cos, sin, pair_matrix).transpose(1, 2)
        key = self._rotate(self.k_proj(hidden_states).view(shape), cos, sin, pair_matrix).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        weights = (query @ key.transpose(-1, -2) * self.head_dim**-0.5).softmax(dim=-1)
        attended = (weights @ value).transpose(1, 2).reshape(hidden_states.shape)
        return self.o_proj(attended)


class Llama4VisionLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        self.self_attn = Llama4VisionFullAttention(layer.self_attn)
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp

    def forward(self, hidden_states, cos, sin, pair_matrix):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), cos, sin, pair_matrix)
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Llama4VisionModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.patch_embedding = model.patch_embedding
        self.class_embedding = model.class_embedding
        self.positional_embedding_vlm = model.positional_embedding_vlm
        self.layernorm_pre = model.layernorm_pre
        self.layers = nn.ModuleList(Llama4VisionLayer(layer) for layer in model.model.layers)
        self.layernorm_post = model.layernorm_post
        self.vision_adapter = model.vision_adapter
        # HF stores one complex value per rotated pair; expand to the interleaved per-element layout.
        frequencies = model.rotary_embedding.freqs_ci
        self.register_buffer("cos", frequencies.real.unsqueeze(0).repeat_interleave(2, dim=-1), persistent=False)
        self.register_buffer("sin", frequencies.imag.unsqueeze(0).repeat_interleave(2, dim=-1), persistent=False)
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
        positional_embedding = self.positional_embedding_vlm.to(dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = self.layernorm_pre(hidden_states + positional_embedding)
        for layer in self.layers:
            hidden_states = layer(hidden_states, self.cos, self.sin, self.pair_matrix)
        hidden_states = self.layernorm_post(hidden_states)[:, :-1]
        return self.vision_adapter(hidden_states)


class Llama4RotaryEmbedding(RotaryEmbedding):
    def __init__(self, config, max_seq_len_cached):
        super().__init__(config=config, max_seq_len_cached=max_seq_len_cached)
        # The base table stores split-half cos/sin values. Llama 4 uses the same frequencies but
        # rotates adjacent pairs of head dimensions, so expand to the interleaved layout once here.
        half = self._cos_cached.shape[-1] // 2
        self._cos_cached = self._cos_cached[..., :half].repeat_interleave(2, dim=-1)
        self._sin_cached = self._sin_cached[..., :half].repeat_interleave(2, dim=-1)


class Llama4ForCausalLMWrapper(DecoderOnlyWrapper):
    def get_rotary_emb(self, max_seq_len):
        return Llama4RotaryEmbedding(config=self.config, max_seq_len_cached=max_seq_len)

    def get_rbln_attn_class(self):
        return Llama4Attention

    def get_rbln_layer_class(self):
        return Llama4DecoderLayerRegion if self.rbln_config.nested_compile_region else Llama4DecoderLayer

    def get_rbln_model_class(self):
        return Llama4Model


class Llama4DecoderLayer(DecoderOnlyLayer):
    _MLP_ATTR = ("feed_forward",)


class Llama4DecoderLayerRegion(NestedCompileRegionMixin, Llama4DecoderLayer):
    """Decoder layer with a nested compile region around its forward call."""


class Llama4Model(DecoderOnlyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        chunk_size = getattr(self.config, "attention_chunk_size", None)
        self._chunk_size = (
            chunk_size if uses_chunked_attention(self.config) and self.max_seq_len > chunk_size else None
        )
        self._chunked_layers = (
            {idx for idx, layer_type in enumerate(self.config.layer_types) if layer_type == "chunked_attention"}
            if self._chunk_size is not None
            else set()
        )

    def get_chunked_attention_mask(self, attention_mask: torch.Tensor, cache_position: torch.Tensor) -> torch.Tensor:
        positions = cache_position[:, :1] + torch.arange(cache_position.shape[1], device=cache_position.device)
        key_positions = torch.arange(self.max_seq_len, device=cache_position.device)
        chunk_mask = positions[:, :, None] // self._chunk_size == key_positions // self._chunk_size
        chunk_mask = chunk_mask & (key_positions <= positions[:, :, None])
        return attention_mask * chunk_mask[:, None, :, :].to(attention_mask.dtype)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        query_position: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        rotary_emb: nn.Module | None = None,
        global_block_tables: torch.Tensor | None = None,
        local_block_tables: torch.Tensor | None = None,
        lora_int_id: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.get_embedding()(input_ids)
        hidden_states = inputs_embeds

        position_ids = position_ids if position_ids is not None else cache_position
        cos, sin = rotary_emb(hidden_states, self.max_seq_len)
        cos, sin = slice_and_unsqueeze_cos_sin(cos, sin, position_ids)

        if self.attn_impl == "flash_attn":
            seq_positions = self.convert_sequence_positions_for_flash_attn(cache_position[:, 0], self.max_seq_len)
        else:
            seq_positions = cache_position[:, :1]

        chunked_attention_mask = (
            self.get_chunked_attention_mask(attention_mask, cache_position) if self._chunk_size is not None else None
        )

        all_hidden_states = () if output_hidden_states else None
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=chunked_attention_mask if layer_idx in self._chunked_layers else attention_mask,
                seq_positions=seq_positions,
                past_key_values=past_key_values,
                cos=cos,
                sin=sin,
                block_tables=global_block_tables,
                lora_int_id=lora_int_id,
            )

        hidden_states = self.get_last_layernorm()(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return hidden_states, all_hidden_states


class Llama4Attention(DecoderOnlyAttention):
    def __post_init__(self, self_attn):
        super().__post_init__(self_attn)
        self.use_rope = self_attn.use_rope
        self.qk_norm = getattr(self_attn, "qk_norm", None)
        self.attn_temperature_tuning = self_attn.attn_temperature_tuning
        self.floor_scale = self_attn.floor_scale
        self.attn_scale = self_attn.attn_scale

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        if not self.use_rope:
            return query_states, key_states

        # Real-valued form of HF's complex RoPE; low-precision results can differ slightly.
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

        if self.attn_temperature_tuning and not self.use_rope:
            positions = seq_positions[:, :1] + torch.arange(query_length, device=hidden_states.device)
            attn_scales = torch.log1p(torch.floor((positions.float() + 1.0) / self.floor_scale))
            attn_scales = (attn_scales * self.attn_scale + 1.0)[:, None, :, None]
            query_states = (query_states * attn_scales).to(query_states.dtype)

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
