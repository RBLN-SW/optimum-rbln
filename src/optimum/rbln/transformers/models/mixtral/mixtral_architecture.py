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
    _MLP_ATTR = None

    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config: Optional[RBLNLoRAConfig] = None):
        super().__init__(layer, self_attn, lora_config)
        self.mlp = MixtralSparseMoeBlock(layer.block_sparse_moe)


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.top_k = model.top_k
        self.gate = model.gate
        self.experts = MixtralBlockSparseTop2MLP(model.experts, self.top_k)
        model.experts = None

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
        self.top_k = top_k

        self.w1_weight = nn.Parameter(torch.stack([expert.w1.weight.data for expert in expert_list], dim=0))
        self.w2_weight = nn.Parameter(torch.stack([expert.w2.weight.data for expert in expert_list], dim=0))
        self.w3_weight = nn.Parameter(torch.stack([expert.w3.weight.data for expert in expert_list], dim=0))

    def forward(self, x, router_logits):
        return torch.ops.rbln_custom_ops.custom_moe_glu(
            x, self.w1_weight, self.w3_weight, self.w2_weight, router_logits, self.top_k, True
        )
