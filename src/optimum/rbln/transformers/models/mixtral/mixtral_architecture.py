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

from typing import Optional

import torch
from torch import nn

from ..decoderonly.configuration_decoderonly import RBLNLoRAConfig
from ..decoderonly.decoderonly_architecture import DecoderOnlyAttention, DecoderOnlyLayer, DecoderOnlyWrapper


class MixtralWrapper(DecoderOnlyWrapper):
    def get_rbln_layer_class(self):
        return MixtralLayer


class MixtralLayer(DecoderOnlyLayer):
    _MLP_ATTR = ("block_sparse_moe",)

    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config: Optional[RBLNLoRAConfig] = None):
        super().__init__(layer, self_attn, lora_config)
        self.block_sparse_moe = MixtralSparseMoeBlock(self.mlp)

    def get_mlp(self) -> nn.Module:
        return self.block_sparse_moe


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        # self.num_experts = model.num_experts
        self.top_k = model.top_k
        self.gate = model.gate
        self.experts = MixtralBlockSparseTop2MLP(model.experts, self.top_k)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, expert_list, top_k):
        super().__init__()
        self.hidden_dim = expert_list[0].hidden_dim
        self.ffn_dim = expert_list[0].ffn_dim
        self.top_k = top_k

        self.num_experts = len(expert_list)
        self.w1 = nn.Linear(self.hidden_dim, self.num_experts * self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.num_experts * self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.num_experts * self.ffn_dim, bias=False)
        self.w1.weight.data = torch.stack([expert.w1.weight.data for expert in expert_list], dim=0)
        self.w2.weight.data = torch.stack([expert.w2.weight.data for expert in expert_list], dim=0)
        self.w3.weight.data = torch.stack([expert.w3.weight.data for expert in expert_list], dim=0)

    def forward(self, x, router_logits):
        return torch.ops.rbln_custom_ops.custom_moe_glu(
            x, self.w1.weight, self.w3.weight, self.w2.weight, router_logits, self.top_k, True
        )
