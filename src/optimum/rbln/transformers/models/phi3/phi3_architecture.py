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


class Phi3Wrapper(DecoderOnlyWrapper):
    def get_rbln_attn_class(self):
        return Phi3Attention

    def get_rbln_layer_class(self):
        return Phi3Layer


def _split_rows(linear: nn.Linear, sizes: list[int]) -> list[nn.Linear]:
    splits = []
    for rows in torch.split(linear.weight.data, sizes, dim=0):
        part = nn.Linear(linear.in_features, rows.shape[0], bias=False)
        part.weight = nn.Parameter(rows.contiguous(), requires_grad=False)
        splits.append(part)
    return splits


class Phi3Attention(DecoderOnlyAttention):
    def __post_init__(self, self_attn):
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        self.q_proj, self.k_proj, self.v_proj = _split_rows(self_attn.qkv_proj, [q_size, kv_size, kv_size])
        self.o_proj = self_attn.o_proj
        self.rotary_ndims = getattr(self.config, "rope_parameters", {}).get("partial_rotary_factor", 1.0)
        self.rotary_ndims = int(self.head_dim * self.rotary_ndims)

        if self.lora_config:
            self._init_lora_weights()

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        if self.rotary_ndims == self.head_dim:
            return super().apply_rotary_pos_embed(query_states, key_states, cos, sin)
        return apply_rotary_pos_emb_partial(query_states, key_states, cos, sin, ndim=self.rotary_ndims)


class Phi3SplitMLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        intermediate_size = mlp.down_proj.in_features
        self.gate_proj, self.up_proj = _split_rows(mlp.gate_up_proj, [intermediate_size, intermediate_size])
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.activation_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Phi3Layer(DecoderOnlyLayer):
    def __init__(self, layer, self_attn, lora_config=None):
        layer.mlp = Phi3SplitMLP(layer.mlp)
        super().__init__(layer, self_attn, lora_config=lora_config)
