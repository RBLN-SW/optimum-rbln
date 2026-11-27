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

from ...utils.moe import compute_masked_routing_weight_softmax_first, split_fused_experts
from ..decoderonly.configuration_lora import RBLNLoRAConfig
from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    NestedCompileRegionMixin,
)
from ..qwen3_5.qwen3_5_architecture import (
    Qwen3_5_CausalLMWrapper,
    Qwen3_5_LanguageModelWrapper,
    Qwen3_5GatedDeltaNet,
    Qwen3_5LinearDecoderLayer,
)


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """HF `Qwen3_5MoeSparseMoeBlock` on the RBLN `custom_moe` op.

    The routed experts run as one fused op over a dense `[E, T]` routing matrix (zero where a token
    did not pick the expert); the shared expert and its sigmoid gate are the HF modules, added as in HF.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.num_experts = model.gate.num_experts
        self.top_k = model.gate.top_k
        gate_weight = model.gate.weight
        gate = nn.Linear(gate_weight.shape[1], gate_weight.shape[0], bias=False)
        gate.weight = model.gate.weight
        self.gate = gate
        self.shared_expert = model.shared_expert
        self.shared_expert_gate = model.shared_expert_gate
        self.experts = Qwen3_5MoeMLP(model.experts, self.top_k)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits)
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            torch.nn.functional.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )
        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class Qwen3_5MoeMLP(nn.Module):
    def __init__(self, experts: nn.Module, top_k: int):
        super().__init__()
        self.num_experts = experts.num_experts
        self.top_k = top_k
        # Qwen3_5MoeTopKRouter always divides the top-k probabilities by their sum.
        self.norm_topk_prob = True

        # Fused Qwen3_5MoeExperts: gate_up_proj [E, 2I, H], down_proj [E, H, I].
        gate, up, down = split_fused_experts(experts)
        self.register_buffer("gate_proj_weight", gate)
        self.register_buffer("up_proj_weight", up)
        self.register_buffer("down_proj_weight", down)

    def forward(self, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        masked_routing_weight = compute_masked_routing_weight_softmax_first(
            router_logits, top_k=self.top_k, renormalize=self.norm_topk_prob
        )
        return torch.ops.rbln_custom_ops.custom_moe(
            hidden_states=x,
            gate_proj_weight=self.gate_proj_weight,
            up_proj_weight=self.up_proj_weight,
            down_proj_weight=self.down_proj_weight,
            masked_routing_weight=masked_routing_weight,
            hidden_act="silu",
        )


def _wrap_mlp(mlp: nn.Module) -> nn.Module:
    return Qwen3_5MoeSparseMoeBlock(mlp) if mlp.__class__.__name__ == "Qwen3_5MoeSparseMoeBlock" else mlp


class Qwen3_5MoeLayer(DecoderOnlyLayer):
    """A `full_attention` layer whose MLP is the sparse MoE block."""

    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config: RBLNLoRAConfig | None = None):
        super().__init__(layer, self_attn, lora_config)
        self.mlp = _wrap_mlp(layer.mlp)

    def get_mlp(self) -> nn.Module:
        return self.mlp


class Qwen3_5MoeLayerRegion(NestedCompileRegionMixin, Qwen3_5MoeLayer):
    """Decoder layer with a nested compile region around its forward call."""


class Qwen3_5MoeLinearDecoderLayer(Qwen3_5LinearDecoderLayer):
    """A `linear_attention` (GatedDeltaNet) layer whose MLP is the sparse MoE block."""

    def __init__(self, layer: nn.Module, linear_attn: Qwen3_5GatedDeltaNet):
        super().__init__(layer, linear_attn)
        self.mlp = _wrap_mlp(layer.mlp)


class Qwen3_5MoeLinearDecoderLayerRegion(NestedCompileRegionMixin, Qwen3_5MoeLinearDecoderLayer):
    """Linear attention decoder layer with a nested compile region around its forward call."""


class _Qwen3_5MoeLayerClasses:
    """Picks the MoE layer classes for both layer kinds; mixed in ahead of the dense wrappers."""

    def get_rbln_layer_class(self):
        return Qwen3_5MoeLayerRegion if self.rbln_config.nested_compile_region else Qwen3_5MoeLayer

    def get_rbln_linear_layer_class(self):
        if self.rbln_config.nested_compile_region:
            return Qwen3_5MoeLinearDecoderLayerRegion
        return Qwen3_5MoeLinearDecoderLayer


class Qwen3_5Moe_CausalLMWrapper(_Qwen3_5MoeLayerClasses, Qwen3_5_CausalLMWrapper):
    """The text-only hybrid decoder with MoE MLPs."""


class Qwen3_5Moe_LanguageModelWrapper(_Qwen3_5MoeLayerClasses, Qwen3_5_LanguageModelWrapper):
    """The hybrid text backbone with MoE MLPs, wired for the vision-language runtime."""
