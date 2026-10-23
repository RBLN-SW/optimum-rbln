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

from ...utils.moe import compute_masked_routing_weight_softmax_first
from ..decoderonly.configuration_decoderonly import RBLNLoRAConfig
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
        # Keep only the compiled portion, as in the Qwen3-VL vision wrapper.
        # Patch embedding runs on CPU; the complex rotary table is not part of
        # the exported module, only its real-valued cos/sin lookup tables are.
        self.class_embedding = model.class_embedding
        self.positional_embedding_vlm = model.positional_embedding_vlm
        self.layernorm_pre = model.layernorm_pre
        self.layers = model.model.layers
        self.layernorm_post = model.layernorm_post
        self.vision_adapter = model.vision_adapter
        frequencies = model.rotary_embedding.freqs_ci
        self.register_buffer("cos", frequencies.real.unsqueeze(0), persistent=False)
        self.register_buffer("sin", frequencies.imag.unsqueeze(0), persistent=False)
        num_pairs = frequencies.shape[-1]
        indices = torch.arange(2 * num_pairs)
        self.register_buffer("interleave_indices", indices // 2 + indices % 2 * num_pairs, persistent=False)

    def _rotate(self, states):
        pairs = states.float().reshape(*states.shape[:-1], -1, 2)
        real, imag = pairs[..., 0], pairs[..., 1]
        out_real = real * self.cos - imag * self.sin
        out_imag = real * self.sin + imag * self.cos
        rotated = torch.cat((out_real, out_imag), dim=-1).index_select(-1, self.interleave_indices)
        return rotated.to(states.dtype)

    def forward(self, hidden_states):
        cls_embedding = self.class_embedding.expand(hidden_states.shape[0], 1, -1)
        hidden_states = torch.cat((hidden_states, cls_embedding), dim=1)
        hidden_states = self.layernorm_pre(hidden_states + self.positional_embedding_vlm)
        for layer in self.layers:
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            attention = layer.self_attn
            shape = (*normalized.shape[:-1], attention.num_heads, attention.head_dim)
            query = self._rotate(attention.q_proj(normalized).view(shape)).transpose(1, 2)
            key = self._rotate(attention.k_proj(normalized).view(shape)).transpose(1, 2)
            value = attention.v_proj(normalized).view(shape).transpose(1, 2)
            weights = (query @ key.transpose(-1, -2) * attention.head_dim**-0.5).softmax(dim=-1)
            attended = (weights @ value).transpose(1, 2).reshape(normalized.shape)
            hidden_states = residual + attention.o_proj(attended)
            hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
        hidden_states = self.layernorm_post(hidden_states)[:, :-1]
        return self.vision_adapter(hidden_states)


class Llama4Wrapper(DecoderOnlyWrapper):
    def get_rotary_emb(self, max_seq_len):
        # The shared lookup table stores split-half cos/sin values. Llama 4 uses
        # the same frequencies, but rotates adjacent pairs of head dimensions.
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

    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config: RBLNLoRAConfig | None = None):
        super().__init__(layer, self_attn, lora_config)
        self.mlp = (
            Llama4TextMoe(layer.feed_forward)
            if layer.feed_forward.__class__.__name__ == "Llama4TextMoe"
            else layer.feed_forward
        )

    def get_lora_linear_class(self):
        return Llama4LoRALinear


class Llama4TextMoe(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        if model.top_k != 1:
            raise NotImplementedError("Llama 4 MoE supports only `num_experts_per_tok=1`.")
        router_weight = model.router.weight
        self.router = nn.Linear(router_weight.shape[1], router_weight.shape[0], bias=False)
        self.router.weight = router_weight
        self.shared_expert = model.shared_expert
        self.experts = Llama4TextExperts(model.experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, shape[-1])
        router_logits = self.router(hidden_states)
        masked_routing_weight = compute_masked_routing_weight_softmax_first(router_logits, top_k=1, renormalize=True)
        routing_scores = torch.sigmoid(router_logits.amax(dim=-1, keepdim=True).float()).to(hidden_states.dtype)
        # Llama 4 weights expert inputs; top-1 permits one scaled input per token.
        routed_states = hidden_states * routing_scores
        routed_output = self.experts(routed_states, masked_routing_weight)
        return (self.shared_expert(hidden_states) + routed_output).reshape(shape)


class Llama4TextExperts(nn.Module):
    def __init__(self, experts: nn.Module):
        super().__init__()
        # HF Llama 4 stores [E, H, 2I] and [E, I, H]; the SDK takes linear weights.
        gate, up = experts.gate_up_proj.detach().chunk(2, dim=-1)
        self.register_buffer("gate_proj_weight", gate.transpose(1, 2).contiguous())
        self.register_buffer("up_proj_weight", up.transpose(1, 2).contiguous())
        self.register_buffer("down_proj_weight", experts.down_proj.detach().transpose(1, 2).contiguous())

    def forward(self, hidden_states: torch.Tensor, masked_routing_weight: torch.Tensor) -> torch.Tensor:
        return torch.ops.rbln_custom_ops.custom_moe_glu(
            hidden_states=hidden_states,
            gate_proj_weight=self.gate_proj_weight,
            up_proj_weight=self.up_proj_weight,
            down_proj_weight=self.down_proj_weight,
            masked_routing_weight=masked_routing_weight,
            hidden_act="silu",
        )


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

        if batch_size > 1 and "prefill" in self.phase:
            raise NotImplementedError(f"batch size should be 1 if prefill phase, but got {batch_size}.")

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
