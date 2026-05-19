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


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.top_k = model.top_k
        # v4 gate is nn.Linear (returns logits tensor); v5 gate is
        # MixtralTopKRouter (returns a tuple) but exposes the same [E, H]
        # weight. Materialize a plain nn.Linear so downstream stays uniform.
        gate_weight = model.gate.weight
        gate = nn.Linear(gate_weight.shape[1], gate_weight.shape[0], bias=False)
        gate.weight = nn.Parameter(gate_weight.detach().clone())
        self.gate = gate
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
    def __init__(self, experts, top_k):
        super().__init__()
        self.top_k = top_k

        if hasattr(experts, "gate_up_proj"):
            # v5: experts is a single MixtralExperts module with fused tensors
            # gate_up_proj [E, 2I, H] (gate || up along the row axis) and
            # down_proj [E, H, I].
            gate_up = experts.gate_up_proj.data
            intermediate_size = gate_up.shape[1] // 2
            self.w1_weight = nn.Parameter(gate_up[:, :intermediate_size, :].contiguous())
            self.w3_weight = nn.Parameter(gate_up[:, intermediate_size:, :].contiguous())
            self.w2_weight = nn.Parameter(experts.down_proj.data.contiguous())
        else:
            # v4: experts is a ModuleList of MixtralBlockSparseTop2MLP, each
            # with per-expert nn.Linear w1/w2/w3.
            self.w1_weight = nn.Parameter(torch.stack([e.w1.weight.data for e in experts], dim=0))
            self.w2_weight = nn.Parameter(torch.stack([e.w2.weight.data for e in experts], dim=0))
            self.w3_weight = nn.Parameter(torch.stack([e.w3.weight.data for e in experts], dim=0))

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
