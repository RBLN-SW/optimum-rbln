# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn as nn

from ....ops.moe import compute_masked_routing_weight_softmax_first
from ..decoderonly.configuration_lora import RBLNLoRAConfig
from ..decoderonly.decoderonly_architecture import DecoderOnlyAttention, DecoderOnlyLayer
from ..qwen3_vl.qwen3_vl_architecture import (
    Qwen3VL_LanguageModelWrapper,
    Qwen3VLAttention,
    Qwen3VLDecoderOnlyForCausalLM,
    Qwen3VLDecoderOnlyModel,
    Qwen3VLVisionBlock,
    Qwen3VLVisionModelWrapper,
)


class Qwen3VLMoeVisionModelWrapper(Qwen3VLVisionModelWrapper):
    pass


class Qwen3VLMoeVisionBlock(Qwen3VLVisionBlock):
    pass


class Qwen3VLMoeAttention(Qwen3VLAttention):
    pass


class Qwen3VLMoeLayer(DecoderOnlyLayer):
    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config: Optional[RBLNLoRAConfig] = None):
        super().__init__(layer, self_attn, lora_config)
        self.mlp = (
            Qwen3VLMoeSparseMoeBlock(layer.mlp)
            if layer.mlp.__class__.__name__ == "Qwen3VLMoeTextSparseMoeBlock"
            else layer.mlp
        )

    def get_mlp(self) -> nn.Module:
        return self.mlp


class Qwen3VLMoeSparseMoeBlock(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.num_experts = model.gate.num_experts
        self.top_k = model.gate.top_k
        gate_weight = model.gate.weight
        gate = nn.Linear(gate_weight.shape[1], gate_weight.shape[0], bias=False)
        gate.weight = nn.Parameter(gate_weight.detach().clone())
        self.gate = gate
        self.experts = Qwen3VLMoeMLP(model.experts, self.top_k)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class Qwen3VLMoeMLP(nn.Module):
    def __init__(self, experts: nn.Module, top_k: int):
        super().__init__()
        self.num_experts = experts.num_experts
        self.top_k = top_k
        self.norm_topk_prob = True

        self.gate_proj = nn.Linear(1, 1, bias=False)
        self.up_proj = nn.Linear(1, 1, bias=False)
        self.down_proj = nn.Linear(1, 1, bias=False)

        # Fused Qwen3VLMoeTextExperts: gate_up_proj [E, 2I, H], down_proj [E, H, I].
        intermediate_dim = experts.intermediate_dim
        gate_up = experts.gate_up_proj.detach().clone()
        self.gate_proj.weight = nn.Parameter(gate_up[:, :intermediate_dim, :].contiguous())
        self.up_proj.weight = nn.Parameter(gate_up[:, intermediate_dim:, :].contiguous())
        self.down_proj.weight = nn.Parameter(experts.down_proj.detach().clone().contiguous())

    def forward(self, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        masked_routing_weight = compute_masked_routing_weight_softmax_first(
            router_logits, top_k=self.top_k, renormalize=self.norm_topk_prob
        )
        return torch.ops.rbln_custom_ops.custom_moe_glu(
            hidden_states=x,
            gate_proj_weight=self.gate_proj.weight,
            up_proj_weight=self.up_proj.weight,
            down_proj_weight=self.down_proj.weight,
            masked_routing_weight=masked_routing_weight,
            hidden_act="silu",
        )


class Qwen3VLMoeDecoderOnlyModel(Qwen3VLDecoderOnlyModel):
    pass


class Qwen3VLMoeDecoderOnlyForCausalLM(Qwen3VLDecoderOnlyForCausalLM):
    pass


class Qwen3VLMoe_LanguageModelWrapper(Qwen3VL_LanguageModelWrapper):
    def get_rbln_layer_class(self):
        return Qwen3VLMoeLayer

    def get_rbln_attn_class(self):
        return Qwen3VLMoeAttention

    def get_rbln_model_class(self):
        return Qwen3VLMoeDecoderOnlyModel

    def get_rbln_causal_lm_class(self):
        return Qwen3VLMoeDecoderOnlyForCausalLM
