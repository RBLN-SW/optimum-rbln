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
from torch import Tensor


@torch.library.custom_op(
    "rbln_custom_ops::custom_moe_glu",
    mutates_args=(),
)
def custom_moe_glu(
    hidden_states: Tensor,
    gate_proj_weight: Tensor,
    up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    masked_routing_weight: Tensor,
    expert_select_count: Tensor,
    gate_proj_bias: Optional[Tensor] = None,
    up_proj_bias: Optional[Tensor] = None,
    down_proj_bias: Optional[Tensor] = None,
) -> Tensor:
    """
    Customized MoE GLU operation.

    Expected tensor shapes:
    - hidden_states: [batch*seq_len, hidden_size]
    - gate_proj_weight: [num_experts, hidden_size, intermediate_size]
    - up_proj_weight: [num_experts, hidden_size, intermediate_size]
    - down_proj_weight: [num_experts, intermediate_size, hidden_size]
    - masked_routing_weight: [batch*seq_len, num_experts]
    - expert_select_count: [num_experts]
    - gate_proj_bias: [num_experts, intermediate_size]
    - up_proj_bias: [num_experts, intermediate_size]
    - down_proj_bias: [num_experts, hidden_size]

    Returns:
        Tensor: [batch * seq_len, hidden_size]
    """

    return torch.empty_like(hidden_states)


@custom_moe_glu.register_fake
def custom_moe_glu_fake(
    hidden_states: Tensor,
    gate_proj_weight: Tensor,
    up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    masked_routing_weight: Tensor,
    expert_select_count: Tensor,
    gate_proj_bias: Optional[Tensor] = None,
    up_proj_bias: Optional[Tensor] = None,
    down_proj_bias: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(hidden_states)


@torch.library.custom_op(
    "rbln_custom_ops::custom_moe_ff",
    mutates_args=(),
)
def custom_moe_ff(
    hidden_states: Tensor,
    gate_proj_weight: Tensor,
    down_proj_weight: Tensor,
    masked_routing_weight: Tensor,
    gate_proj_bias: Optional[Tensor] = None,
    down_proj_bias: Optional[Tensor] = None,
) -> Tensor:
    """
    Customized MoE FF operation.

    Expected tensor shapes:
    - hidden_states: [batch * seq_len, hidden_size]
    - gate_proj_weight: [hidden_size, num_experts * intermediate_size]
    - down_proj_weight: [num_experts * intermediate_size, hidden_size]
    - masked_routing_weight: [batch * seq_len, num_experts]
    - gate_proj_bias: [num_experts * intermediate_size]
    - down_proj_bias: [hidden_size]

    Returns:
        Tensor: [batch * seq_len, hidden_size]
    """
    return torch.empty_like(hidden_states)


@custom_moe_ff.register_fake
def custom_moe_ff_fake(
    hidden_states: Tensor,
    gate_proj_weight: Tensor,
    down_proj_weight: Tensor,
    masked_routing_weight: Tensor,
    gate_proj_bias: Optional[Tensor] = None,
    down_proj_bias: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(hidden_states)


@torch.library.custom_op(
    "rbln_custom_ops::custom_moe_glu_mxfp4",
    mutates_args=(),
)
def custom_moe_glu_mxfp4(
    hidden_states: Tensor,
    gate_proj_blocks: Tensor,
    gate_proj_scales: Tensor,
    gate_proj_bias: Tensor,
    up_proj_blocks: Tensor,
    up_proj_scales: Tensor,
    up_proj_bias: Tensor,
    down_proj_blocks: Tensor,
    down_proj_scales: Tensor,
    down_proj_bias: Tensor,
    router_logits: Tensor,
    alpha: Tensor,
    limit: Tensor,
    k: int,
    post_norm: bool = True,
) -> Tensor:
    """
    Customized MoE GLU operation.

    Expected tensor shapes:
    - hidden_states: [batch*seq_len, hidden_size]
    - gate_proj_blocks: [num_experts, intermediate_size, hidden_size // 2]
    - gate_proj_scales: [num_experts, intermediate_size, hidden_size // 32]
    - gate_proj_bias: [num_experts, intermediate_size]
    - up_proj_blocks: [num_experts, intermediate_size, hidden_size // 2]
    - up_proj_scales: [num_experts, intermediate_size, hidden_size // 32]
    - up_proj_bias: [num_experts, intermediate_size]
    - down_proj_blocks: [num_experts, hidden_size, intermediate_size // 2]
    - down_proj_scales: [num_experts, hidden_size, intermediate_size // 32]
    - masked_routing_weight: [batch * seq_len, num_experts]
    - expert_select_count: [num_experts]
    - alpha: []
    - limit: []

    Returns:
        Tensor: [batch * seq_len, hidden_size]
    """

    return torch.empty_like(hidden_states)


@custom_moe_glu_mxfp4.register_fake
def custom_moe_glu_mxfp4_fake(
    hidden_states: Tensor,
    gate_proj_blocks: Tensor,
    gate_proj_scales: Tensor,
    gate_proj_bias: Tensor,
    up_proj_blocks: Tensor,
    up_proj_scales: Tensor,
    up_proj_bias: Tensor,
    down_proj_blocks: Tensor,
    down_proj_scales: Tensor,
    down_proj_bias: Tensor,
    router_logits: Tensor,
    alpha: Tensor,
    limit: Tensor,
    k: int,
    post_norm: bool = True,
) -> Tensor:
    return torch.empty_like(hidden_states)
