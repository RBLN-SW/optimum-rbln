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
import torch.nn.functional as F
from torch import nn

from ..decoderonly.configuration_decoderonly import RBLNLoRAConfig
from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyWrapper,
)


class RBLNGptOssWrapper(DecoderOnlyWrapper):
    def get_rbln_attn_class(self):
        return RBLNGptOssAttention

    def get_rbln_layer_class(self):
        return RBLNGptOssLayer


class RBLNGptOssAttention(DecoderOnlyAttention):
    def __post_init__(self):
        # Initialize LoRA weights if configured, which will replace linear layers
        if self.lora_config:
            self._init_lora_weights()
        else:
            # Use original linear layers if no LoRA
            self.q_proj = self._original_mod.q_proj
            self.k_proj = self._original_mod.k_proj
            self.v_proj = self._original_mod.v_proj
            self.o_proj = self._original_mod.o_proj
            self.sinks = self._original_mod.sinks.data[:, None]


class RBLNGptOssLayer(DecoderOnlyLayer):
    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config: Optional[RBLNLoRAConfig] = None):
        super().__init__(layer, self_attn, lora_config)
        self.mlp = RBLNGptOssMLP(layer.mlp)

    def get_mlp(self) -> nn.Module:
        return self.mlp


class RBLNGptOssTopKRouter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.weight = model.weight
        self.bias = model.bias

    def forward(self, hidden_states):
        return F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)


class RBLNGptOssExperts(nn.Module):
    def __init__(self, model, top_k: Optional[int] = None):
        super().__init__()
        self.intermediate_size = model.intermediate_size
        self.num_experts = model.num_experts
        self.hidden_size = model.hidden_size

        self.register_buffer(
            "gate_proj_blocks",
            model.gate_up_proj_blocks.data[:, ::2, :, :].reshape(self.num_experts, self.intermediate_size, -1),
        )
        self.register_buffer("gate_proj_scales", model.gate_up_proj_scales.data[:, ::2, :])
        self.register_buffer(
            "gate_proj_bias",
            model.gate_up_proj_bias.data[:, ::2].reshape(self.num_experts, self.intermediate_size),
        )

        self.register_buffer(
            "up_proj_blocks",
            model.gate_up_proj_blocks.data[:, 1::2, :, :].reshape(self.num_experts, self.intermediate_size, -1),
        )
        self.register_buffer("up_proj_scales", model.gate_up_proj_scales.data[:, 1::2, :])
        self.register_buffer(
            "up_proj_bias", model.gate_up_proj_bias.data[:, 1::2].reshape(self.num_experts, self.intermediate_size)
        )

        self.register_buffer(
            "down_proj_blocks", model.down_proj_blocks.data.reshape(self.num_experts, self.hidden_size, -1)
        )
        self.register_buffer("down_proj_scales", model.down_proj_scales.data)
        self.register_buffer("down_proj_bias", model.down_proj_bias.data)

        self.alpha = model.alpha  # 1.702
        self.limit = model.limit  # 7.0
        self.top_k = top_k

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        return torch.ops.rbln_custom_ops.custom_moe_glu_mxfp4(
            hidden_states,
            self.gate_proj_blocks,
            self.gate_proj_scales,
            self.gate_proj_bias,
            self.up_proj_blocks,
            self.up_proj_scales,
            self.up_proj_bias,
            self.down_proj_blocks,
            self.down_proj_scales,
            self.down_proj_bias,
            router_logits,
            torch.tensor(self.alpha, dtype=hidden_states.dtype),
            torch.tensor(self.limit, dtype=hidden_states.dtype),
            k=self.top_k,
        )


class RBLNGptOssMLP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._original_mod = model
        self.router = RBLNGptOssTopKRouter(model.router)
        self.experts = RBLNGptOssExperts(model.experts, top_k=model.router.top_k)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.router(hidden_states)
        routed_out = self.experts(hidden_states, router_logits=router_logits)
        routed_out = routed_out.reshape(batch_size, sequence_length, hidden_dim)
        return routed_out
