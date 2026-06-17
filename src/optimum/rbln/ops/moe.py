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

# The ``rbln_custom_ops::custom_moe_*`` operators are registered by rebel-compiler
# (``import rebel``); see ``optimum/rbln/ops/__init__.py``. This module keeps only the
# routing-weight helpers that the MoE model architectures call before invoking the ops.

import torch
from torch import Tensor


def compute_masked_routing_weight_softmax_first(router_logits: Tensor, top_k: int, renormalize: bool) -> Tensor:
    #   renormalize=True : topk → softmax-of-topk → scatter  (post_norm)
    #   renormalize=False: softmax → topk → scatter           (pre_norm)
    router_logits_t = router_logits.transpose(0, 1)  # [T, E] -> [E, T]
    if renormalize:
        topk_values, topk_ids = torch.topk(router_logits_t, top_k, dim=0)
        topk_weights = torch.softmax(topk_values, dim=0)
    else:
        routing = torch.softmax(router_logits_t, dim=0)
        topk_weights, topk_ids = torch.topk(routing, top_k, dim=0)
    masked = torch.zeros_like(router_logits_t, dtype=router_logits.dtype)
    masked.scatter_(0, topk_ids, topk_weights)
    return masked  # [E, T]


def compute_masked_routing_weight_topk_first(router_logits: Tensor, top_k: int) -> Tensor:
    # topk → softmax on topk values → scatter (GPT-OSS style).
    # Same [E, T] / dim=0 layout as softmax_first for compiler pattern matching.
    router_logits_t = router_logits.transpose(0, 1)  # [T, E] -> [E, T]
    topk_values, topk_ids = torch.topk(router_logits_t, top_k, dim=0)
    topk_weights = torch.softmax(topk_values, dim=0)
    masked = torch.zeros_like(router_logits_t, dtype=router_logits.dtype)
    masked.scatter_(0, topk_ids, topk_weights)
    return masked  # [E, T]
