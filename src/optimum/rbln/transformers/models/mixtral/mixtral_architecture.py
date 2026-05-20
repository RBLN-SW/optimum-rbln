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
        # transformers v4: layer.block_sparse_moe; v5: layer.mlp.
        moe = getattr(layer, "block_sparse_moe", None) or layer.mlp
        self.mlp = MixtralSparseMoeBlock(moe)


class _MixtralV5ExpertView(nn.Module):
    """v4-shaped (w1/w2/w3 nn.Linear) view onto a single expert sliced out of v5's
    fused MixtralExperts (gate_up_proj [E, 2I, H], down_proj [E, H, I])."""

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor):
        super().__init__()
        out_features, in_features = w1.shape
        self.w1 = nn.Linear(in_features, out_features, bias=False)
        self.w1.weight = nn.Parameter(w1)
        out_features, in_features = w2.shape
        self.w2 = nn.Linear(in_features, out_features, bias=False)
        self.w2.weight = nn.Parameter(w2)
        out_features, in_features = w3.shape
        self.w3 = nn.Linear(in_features, out_features, bias=False)
        self.w3.weight = nn.Parameter(w3)


def _v5_experts_to_v4_list(experts: nn.Module) -> nn.ModuleList:
    """Decompose v5's fused MixtralExperts back into a v4-style per-expert ModuleList
    so the rest of the wrapper can take the same code path on both versions."""
    gate_up = experts.gate_up_proj.detach().clone()
    down = experts.down_proj.detach().clone()
    num_experts = gate_up.shape[0]
    intermediate_size = gate_up.shape[1] // 2
    return nn.ModuleList(
        [
            _MixtralV5ExpertView(
                w1=gate_up[i, :intermediate_size, :].contiguous(),
                w2=down[i, :, :].contiguous(),
                w3=gate_up[i, intermediate_size:, :].contiguous(),
            )
            for i in range(num_experts)
        ]
    )


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.top_k = model.top_k
        if hasattr(model.experts, "gate_up_proj"):
            # v5 gate is MixtralTopKRouter (returns a tuple); rebuild a plain
            # nn.Linear from its [E, H] weight so the rest of the path stays
            # uniform with v4. v5 fused experts are similarly decomposed into
            # the v4-shaped per-expert ModuleList.
            gate_weight = model.gate.weight
            gate = nn.Linear(gate_weight.shape[1], gate_weight.shape[0], bias=False)
            gate.weight = nn.Parameter(gate_weight.detach().clone())
            self.gate = gate
            experts = _v5_experts_to_v4_list(model.experts)
        else:
            self.gate = model.gate
            experts = model.experts
        self.experts = MixtralBlockSparseTop2MLP(experts, self.top_k)

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
            hidden_states=x,
            gate_proj_weight=self.w1_weight,
            up_proj_weight=self.w3_weight,
            down_proj_weight=self.w2_weight,
            router_logits=router_logits,
            scoring_func="softmax",
            topk=self.top_k,
            norm_topk_prob=True,
        )
