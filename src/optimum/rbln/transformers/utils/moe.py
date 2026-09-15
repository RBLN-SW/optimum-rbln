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

import torch
from torch import Tensor, nn


def compute_masked_routing_weight_softmax_first(router_logits: Tensor, top_k: int, renormalize: bool) -> Tensor:
    #   renormalize=True : topk → softmax-of-topk → scatter  (post_norm)
    #   renormalize=False: softmax → topk → scatter           (pre_norm)
    # HF routes in fp32 (softmax(..., dtype=torch.float)) and casts the weights back to the logits dtype.
    router_logits_t = router_logits.transpose(0, 1).to(torch.float32)  # [T, E] -> [E, T]
    if renormalize:
        topk_values, topk_ids = torch.topk(router_logits_t, top_k, dim=0)
        topk_weights = torch.softmax(topk_values, dim=0)
    else:
        routing = torch.softmax(router_logits_t, dim=0)
        topk_weights, topk_ids = torch.topk(routing, top_k, dim=0)
    masked = torch.zeros_like(router_logits_t)
    masked.scatter_(0, topk_ids, topk_weights)
    return masked.to(router_logits.dtype)  # [E, T]


def compute_masked_routing_weight_topk_first(router_logits: Tensor, top_k: int) -> Tensor:
    # topk → softmax on topk values → scatter (GPT-OSS style).
    # Same [E, T] / dim=0 layout as softmax_first for compiler pattern matching.
    router_logits_t = router_logits.transpose(0, 1)  # [T, E] -> [E, T]
    topk_values, topk_ids = torch.topk(router_logits_t, top_k, dim=0)
    topk_weights = torch.softmax(topk_values, dim=0)
    masked = torch.zeros_like(router_logits_t, dtype=router_logits.dtype)
    masked.scatter_(0, topk_ids, topk_weights)
    return masked  # [E, T]


def split_fused_experts(experts: nn.Module) -> tuple[Tensor, Tensor, Tensor]:
    # HF packs gate|up along dim 1 of gate_up_proj [E, 2I, H]; custom_moe_glu takes them separately, so the
    # halves are copied to be contiguous while down_proj [E, H, I] is shared. The fused tensor is released
    # afterwards: the HF experts module is not run once wrapped, and keeping it would double host memory.
    gate_up = experts.gate_up_proj.detach()
    intermediate_dim = gate_up.shape[1] // 2
    gate = gate_up[:, :intermediate_dim, :].contiguous()
    up = gate_up[:, intermediate_dim:, :].contiguous()
    experts.gate_up_proj = None
    return gate, up, experts.down_proj.detach()


def release_checkpoint_mmap_(model: nn.Module) -> nn.Module:
    # transformers leaves whatever it did not convert as a view of the safetensors mmap, which keeps the whole
    # checkpoint resident. Copy everything out except the fused experts gate_up_proj: the wrapper splits it into
    # gate/up copies and drops it, so once wrapped no view remains and the mapping is released. Dense models
    # (no gate_up_proj) are left alone; there the mmap is the model itself.
    modules = list(model.modules())
    if not any("gate_up_proj" in module._parameters for module in modules):
        return model
    for module in modules:
        for name, param in module.named_parameters(recurse=False):
            if name != "gate_up_proj":
                param.data = param.data.clone()
        # Buffers are installed as the mmap view itself (no Parameter wrap), and a view keeps its base alive
        # through `.data` swaps, so replace the buffer object instead.
        for name, buffer in module.named_buffers(recurse=False):
            setattr(module, name, buffer.detach().clone())
    return model


class RBLNMoeLoadMixin:
    @classmethod
    def get_pytorch_model(cls, *args, **kwargs):
        return release_checkpoint_mmap_(super().get_pytorch_model(*args, **kwargs))
