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

import math

import torch
import torch.nn as nn

from .configuration_qwen2_5_omni import RBLNQwen2_5OmniToken2WavDiTModelConfig


def rotate_half_codec(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.reshape(*x.shape[:-2], -1)


def apply_rotary_pos_emb_dit(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half_codec(q) * sin)
    k_embed = (k * cos) + (rotate_half_codec(k) * sin)
    return q_embed, k_embed


class Qwen2_5OmniToken2WavDiTWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        rbln_config: RBLNQwen2_5OmniToken2WavDiTModelConfig,
    ):
        super().__init__()
        self.rbln_config = rbln_config
        self.hidden_size = model.hidden_size
        self.num_attention_heads = model.num_attention_heads
        self.block_size = model.block_size

        # Transformer blocks only - embeddings are handled on CPU
        self.transformer_blocks = nn.ModuleList(
            [DiTDecoderLayerWrapper(block, rbln_config) for block in model.transformer_blocks]
        )

        # Output layers
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_embedding: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_diff: torch.Tensor,
    ) -> torch.Tensor:
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(
                hidden_states,
                time_embedding,
                cos=cos,
                sin=sin,
                block_diff=block_diff,
            )

        hidden_states = self.norm_out(hidden_states, time_embedding)
        output = self.proj_out(hidden_states)

        return output


class DiTDecoderLayerWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        rbln_config: RBLNQwen2_5OmniToken2WavDiTModelConfig,
    ):
        super().__init__()
        self.rbln_config = rbln_config
        self.look_ahead_block = model.look_ahead_block
        self.look_backward_block = model.look_backward_block

        self.attn_norm = model.attn_norm
        self.attn = DiTAttentionWrapper(model.attn, rbln_config)
        self.ff_norm = model.ff_norm
        self.ff = model.ff

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_diff: torch.Tensor,
    ) -> torch.Tensor:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(hidden_states, emb=timestep)

        attention_mask = (block_diff >= -float(self.look_backward_block)) & (
            block_diff <= float(self.look_ahead_block)
        )

        attn_output = self.attn(
            hidden_states=norm,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        return hidden_states


class DiTAttentionWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        rbln_config: RBLNQwen2_5OmniToken2WavDiTModelConfig,
    ):
        super().__init__()
        self.rbln_config = rbln_config
        self.dim = model.dim
        self.heads = model.heads
        self.inner_dim = model.inner_dim
        self.head_dim = self.inner_dim // self.heads
        self.dropout = model.dropout

        # Projection layers
        self.to_q = model.to_q
        self.to_k = model.to_k
        self.to_v = model.to_v
        self.to_out = model.to_out

        # Attention scale
        self.scale = nn.Parameter(
            torch.tensor(1.0 / math.sqrt(self.head_dim), dtype=rbln_config.dtype),
            requires_grad=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # Reshape to [batch, heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embedding (only to first head as per original implementation)
        query[:, :1], key[:, :1] = apply_rotary_pos_emb_dit(query[:, :1], key[:, :1], cos, sin)
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scale

        if attention_mask is not None:
            attn_mask_float = torch.where(
                attention_mask,
                torch.zeros_like(attn_weights),
                torch.full_like(attn_weights, torch.finfo(attn_weights.dtype).min),
            )
            attn_weights = attn_weights + attn_mask_float

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        attn_output = torch.matmul(attn_weights, value)

        # Reshape output [batch, heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.heads * self.head_dim)
        attn_output = self.to_out[0](attn_output)
        attn_output = self.to_out[1](attn_output)

        return attn_output
