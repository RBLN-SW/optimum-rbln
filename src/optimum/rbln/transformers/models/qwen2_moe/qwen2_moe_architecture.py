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
from torch.nn import functional as F

from ..decoderonly.configuration_decoderonly import RBLNLoRAConfig
from ..decoderonly.decoderonly_architecture import DecoderOnlyAttention, DecoderOnlyLayer, DecoderOnlyWrapper


class Qwen2MoeWrapper(DecoderOnlyWrapper):
    def get_rbln_layer_class(self):
        return Qwen2MoeLayer


class Qwen2MoeLayer(DecoderOnlyLayer):
    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config: Optional[RBLNLoRAConfig] = None):
        super().__init__(layer, self_attn, lora_config)
        self.mlp = (
            Qwen2MoeSparseMoeBlock(layer.mlp)
            if layer.mlp.__class__.__name__ == "Qwen2MoeSparseMoeBlock"
            else layer.mlp
        )

    def get_mlp(self) -> nn.Module:
        return self.mlp


class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        # transformers==5 stores experts as fused 3D tensors (Qwen2MoeExperts) and the router as Qwen2MoeTopKRouter.
        cfg = getattr(model, "config", None)

        self.gate = model.gate
        self.shared_expert = model.shared_expert
        self.shared_expert_gate = model.shared_expert_gate

        # Routing hyperparams (prefer module, fall back to config)
        self.num_experts = (
            getattr(self.gate, "num_experts", None)
            or getattr(model, "num_experts", None)
            or getattr(cfg, "num_experts", None)
        )
        self.top_k = (
            getattr(self.gate, "top_k", None)
            or getattr(model, "top_k", None)
            or getattr(cfg, "num_experts_per_tok", None)
        )
        self.norm_topk_prob = getattr(self.gate, "norm_topk_prob", None)
        if self.norm_topk_prob is None:
            self.norm_topk_prob = getattr(model, "norm_topk_prob", None)
        if self.norm_topk_prob is None:
            self.norm_topk_prob = getattr(cfg, "norm_topk_prob", False)

        # Experts weights can be either:
        # - transformers<=4: model.experts is a list/ModuleList of per-expert MLP modules
        # - transformers>=5: model.experts is Qwen2MoeExperts with fused [E, ...] tensors
        self.experts = Qwen2MoeMLP(model.experts, self.top_k, self.norm_topk_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        # transformers>=5 gate is Qwen2MoeTopKRouter which returns (router_probs, topk_scores, topk_idx).
        # Our fused op expects router logits to compute top-k internally, so compute raw logits via gate weight if present.
        if hasattr(self.gate, "weight"):
            router_logits = F.linear(hidden_states, self.gate.weight)
        else:
            router_out = self.gate(hidden_states)
            router_logits = router_out[0] if isinstance(router_out, (tuple, list)) else router_out
        final_hidden_states = self.experts(hidden_states, router_logits)
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            torch.nn.functional.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )
        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class Qwen2MoeMLP(nn.Module):
    def __init__(self, expert_list, top_k, norm_topk_prob):
        super().__init__()
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

        # transformers>=5: experts are stored as fused tensors in Qwen2MoeExperts
        if hasattr(expert_list, "gate_up_proj") and hasattr(expert_list, "down_proj"):
            self.num_experts = int(expert_list.num_experts)
            self.hidden_size = int(expert_list.hidden_dim)
            self.intermediate_size = int(expert_list.intermediate_dim)

            gate_up = expert_list.gate_up_proj  # [E, 2*I, H]
            gate_w, up_w = gate_up.chunk(2, dim=1)  # [E, I, H] each
            down_w = expert_list.down_proj  # [E, H, I]
        else:
            # transformers<=4: per-expert modules
            self.hidden_size = expert_list[0].hidden_size
            self.intermediate_size = expert_list[0].intermediate_size
            self.num_experts = len(expert_list)
            gate_w = torch.stack([expert.gate_proj.weight.data for expert in expert_list], dim=0)
            up_w = torch.stack([expert.up_proj.weight.data for expert in expert_list], dim=0)
            down_w = torch.stack([expert.down_proj.weight.data for expert in expert_list], dim=0)

        self.gate_proj = nn.Linear(self.hidden_size, self.num_experts * self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.num_experts * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.num_experts * self.intermediate_size, self.hidden_size, bias=False)
        # Ensure weights are contiguous (rebel hashing uses `.view()` on params).
        self.gate_proj.weight.data = gate_w.contiguous()
        self.up_proj.weight.data = up_w.contiguous()
        self.down_proj.weight.data = down_w.contiguous()

    def forward(self, x, router_logits):
        return torch.ops.rbln_custom_ops.custom_moe_glu(
            x,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
            router_logits,
            self.top_k,
            self.norm_topk_prob,
        )
