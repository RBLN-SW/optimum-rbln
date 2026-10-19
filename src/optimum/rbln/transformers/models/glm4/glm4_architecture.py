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
    DecoderOnlyWrapper,
    apply_rotary_pos_emb_partial,
)


def _deinterleave_heads(linear: nn.Linear, num_heads: int, head_dim: int, rotary_ndims: int) -> nn.Linear:
    # GLM rotates adjacent (even, odd) pairs of the rotary dims. Moving the even dims ahead of the odd ones inside
    # every head turns that into the usual half-split rotation; q and k share the permutation, so q.k is unchanged.
    order = torch.cat(
        [torch.arange(0, rotary_ndims, 2), torch.arange(1, rotary_ndims, 2), torch.arange(rotary_ndims, head_dim)]
    )
    rows = (torch.arange(num_heads)[:, None] * head_dim + order[None, :]).flatten()
    permuted = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
    permuted.weight = nn.Parameter(linear.weight.data[rows].contiguous(), requires_grad=False)
    if linear.bias is not None:
        permuted.bias = nn.Parameter(linear.bias.data[rows].contiguous(), requires_grad=False)
    return permuted


class Glm4Wrapper(DecoderOnlyWrapper):
    def get_rbln_attn_class(self):
        return Glm4Attention

    def get_rbln_layer_class(self):
        return Glm4DecoderLayer


class Glm4Attention(DecoderOnlyAttention):
    def __post_init__(self, self_attn):
        # The q/k rows are permuted below, which the LoRA deltas would not follow.
        if self.lora_config:
            raise NotImplementedError("LoRA is not supported for GLM-4 yet.")
        super().__post_init__(self_attn)
        rotary_factor = getattr(self.config, "rope_parameters", {}).get("partial_rotary_factor", 1.0)
        self.rotary_ndims = int(self.head_dim * rotary_factor)
        self.q_proj = _deinterleave_heads(self.q_proj, self.num_heads, self.head_dim, self.rotary_ndims)
        self.k_proj = _deinterleave_heads(self.k_proj, self.num_key_value_heads, self.head_dim, self.rotary_ndims)

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        return apply_rotary_pos_emb_partial(query_states, key_states, cos, sin, ndim=self.rotary_ndims)


class Glm4SplitMLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        intermediate_size = mlp.down_proj.in_features
        gate_up = mlp.gate_up_proj.weight.data
        self.gate_proj = nn.Linear(gate_up.shape[1], intermediate_size, bias=False)
        self.up_proj = nn.Linear(gate_up.shape[1], intermediate_size, bias=False)
        self.gate_proj.weight = nn.Parameter(gate_up[:intermediate_size].contiguous(), requires_grad=False)
        self.up_proj.weight = nn.Parameter(gate_up[intermediate_size:].contiguous(), requires_grad=False)
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.activation_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Glm4DecoderLayer(DecoderOnlyLayer):
    _PRE_ATTN_LAYERNORM = ["input_layernorm"]
    _POST_ATTN_LAYERNORM = ["post_attention_layernorm"]

    def __init__(self, layer, self_attn, lora_config=None):
        layer.mlp = Glm4SplitMLP(layer.mlp)
        super().__init__(layer, self_attn, lora_config=lora_config)
        self.post_self_attn_layernorm = layer.post_self_attn_layernorm
        self.post_mlp_layernorm = layer.post_mlp_layernorm

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
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.get_post_attention_layernorm()(hidden_states)
        hidden_states = self.forward_mlp(hidden_states, lora_int_id)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
