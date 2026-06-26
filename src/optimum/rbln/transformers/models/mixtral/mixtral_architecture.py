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

from ....ops.moe import compute_masked_routing_weight_softmax_first
from ..decoderonly.configuration_decoderonly import RBLNLoRAConfig
from ..decoderonly.decoderonly_architecture import DecoderOnlyAttention, DecoderOnlyLayer, DecoderOnlyWrapper


class MixtralWrapper(DecoderOnlyWrapper):
    def get_rbln_layer_class(self):
        return MixtralLayer


class MixtralLayer(DecoderOnlyLayer):
    _MLP_ATTR = None

    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config: Optional[RBLNLoRAConfig] = None):
        super().__init__(layer, self_attn, lora_config)
        self.mlp = MixtralSparseMoeBlock(layer.mlp)


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.top_k = model.top_k
        gate_weight = model.gate.weight
        gate = nn.Linear(gate_weight.shape[1], gate_weight.shape[0], bias=False)
        gate.weight = nn.Parameter(gate_weight.detach().clone())
        self.gate = gate
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
    def __init__(self, experts: nn.Module, top_k: int):
        super().__init__()
        self.top_k = top_k

        # Fused MixtralExperts: gate_up_proj [E, 2I, H], down_proj [E, H, I].
        gate_up = experts.gate_up_proj.detach().clone()
        intermediate_size = gate_up.shape[1] // 2
        self.w1_weight = nn.Parameter(gate_up[:, :intermediate_size, :].contiguous())
        self.w3_weight = nn.Parameter(gate_up[:, intermediate_size:, :].contiguous())
        self.w2_weight = nn.Parameter(experts.down_proj.detach().clone().contiguous())

    def forward(self, x, router_logits):
        masked_routing_weight = compute_masked_routing_weight_softmax_first(
            router_logits, top_k=self.top_k, renormalize=True
        )
        return torch.ops.rbln_custom_ops.custom_moe_glu(
            hidden_states=x,
            gate_proj_weight=self.w1_weight,
            up_proj_weight=self.w3_weight,
            down_proj_weight=self.w2_weight,
            masked_routing_weight=masked_routing_weight,
            hidden_act="silu",
        )
