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

from typing import Optional

import torch
from torch import nn

from ..decoderonly.configuration_decoderonly import RBLNLoRAConfig
from ..decoderonly.decoderonly_architecture import DecoderOnlyAttention, DecoderOnlyLayer, DecoderOnlyWrapper


class Qwen3MoeWrapper(DecoderOnlyWrapper):
    def get_rbln_layer_class(self):
        return Qwen3MoeLayer

    def get_rbln_attn_class(self):
        return Qwen3MoeAttention


class Qwen3MoeAttention(DecoderOnlyAttention):
    def __post_init__(self, self_attn):
        self.q_proj = self_attn.q_proj
        self.k_proj = self_attn.k_proj
        self.v_proj = self_attn.v_proj
        self.o_proj = self_attn.o_proj
        self.q_norm = self_attn.q_norm
        self.k_norm = self_attn.k_norm


class Qwen3MoeLayer(DecoderOnlyLayer):
    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config: Optional[RBLNLoRAConfig] = None):
        super().__init__(layer, self_attn, lora_config)
        self.mlp = (
            Qwen3MoeSparseMoeBlock(layer.mlp)
            if layer.mlp.__class__.__name__ == "Qwen3MoeSparseMoeBlock"
            else layer.mlp
        )

    def get_mlp(self) -> nn.Module:
        return self.mlp


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.num_experts = model.experts.num_experts
        self.top_k = model.gate.top_k
        self.norm_topk_prob = model.gate.norm_topk_prob
        gate_weight = model.gate.weight
        gate = nn.Linear(gate_weight.shape[1], gate_weight.shape[0], bias=False)
        gate.weight = nn.Parameter(gate_weight.detach().clone())
        self.gate = gate
        self.experts = Qwen3MoeMLP(model.experts, self.top_k, self.norm_topk_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits)

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class Qwen3MoeMLP(nn.Module):
    def __init__(self, experts, top_k, norm_topk_prob):
        super().__init__()
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

        # Fused Qwen3MoeExperts: gate_up_proj [E, 2I, H], down_proj [E, H, I].
        self.num_experts = experts.num_experts
        self.hidden_size = experts.hidden_dim
        self.intermediate_size = experts.intermediate_dim
        gate_up = experts.gate_up_proj.detach().clone()
        intermediate_size = gate_up.shape[1] // 2
        gate_stack = gate_up[:, :intermediate_size, :].contiguous()
        up_stack = gate_up[:, intermediate_size:, :].contiguous()
        down_stack = experts.down_proj.detach().clone().contiguous()

        self.gate_proj = nn.Linear(self.hidden_size, self.num_experts * self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.num_experts * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.num_experts * self.intermediate_size, self.hidden_size, bias=False)
        self.gate_proj.weight.data = gate_stack
        self.up_proj.weight.data = up_stack
        self.down_proj.weight.data = down_stack

    def forward(self, x, router_logits):
        return torch.ops.rbln_custom_ops.custom_moe_glu(
            hidden_states=x,
            gate_proj_weight=self.gate_proj.weight,
            up_proj_weight=self.up_proj.weight,
            down_proj_weight=self.down_proj.weight,
            router_logits=router_logits,
            scoring_func="softmax",
            topk=self.top_k,
            norm_topk_prob=self.norm_topk_prob,
        )
