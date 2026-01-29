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

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from optimum.rbln.transformers.models.decoderonly.configuration_lora import (
    RBLNLoRAConfig,
)
from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyWrapper,
    apply_rotary_pos_emb_partial,
)


class MiniMaxM2Wrapper(DecoderOnlyWrapper):
    """
    RBLN decoder-only wrapper for MiniMax-M2.* (MoE + optional QK-norm + partial RoPE).

    Notes:
    - MiniMaxM2 attention applies Q/K RMSNorm *before* reshaping into heads. We override `projection()` to match.
    - MiniMaxM2 uses partial rotary embeddings (`rotary_dim` < `head_dim`), so we use `apply_rotary_pos_emb_partial()`.
    - MiniMaxM2 layers use `block_sparse_moe` instead of `mlp`, so we provide a custom layer wrapper.
    """

    def get_rbln_attn_class(self):
        return MiniMaxM2Attention

    def get_rbln_layer_class(self):
        return MiniMaxM2Layer


class MiniMaxM2Attention(DecoderOnlyAttention):
    """
    MiniMax-M2 attention wrapper.

    The upstream MiniMax implementation does:
      q = q_proj(x); k = k_proj(x); v = v_proj(x)
      if use_qk_norm: q = q_norm(q); k = k_norm(k)     # on flattened [B, T, H]
      reshape to heads
      apply RoPE on first rotary_dim
      attention
      o_proj

    Base `DecoderOnlyAttention` applies q_norm/k_norm after reshaping; we need to apply it before.
    """

    def __post_init__(self, self_attn=None):
        # projections are named like llama/mistral in MiniMax HF code
        self.q_proj = self_attn.q_proj
        self.k_proj = self_attn.k_proj
        self.v_proj = self_attn.v_proj
        self.o_proj = self_attn.o_proj

        if self.config.use_qk_norm:
            self.q_norm_1d = getattr(self_attn, "q_norm", None)
            self.k_norm_1d = getattr(self_attn, "k_norm", None)

        # LoRA replacement is handled by the parent class based on `self.lora_config`.
        if self.lora_config:
            self._init_lora_weights()

    def projection(
        self, hidden_states: torch.Tensor, lora_int_id: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states, key_states, value_states = super().projection(
            hidden_states=hidden_states, lora_int_id=lora_int_id
        )

        # Apply QK norm on flattened tensor before (B,T,heads,head_dim) reshape.
        if self.q_norm_1d is not None and self.k_norm_1d is not None:
            query_states = self.q_norm_1d(query_states)
            key_states = self.k_norm_1d(key_states)

        return query_states, key_states, value_states

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        # Partial rotary (rotary_dim) is encoded by cos/sin's last dim.
        return apply_rotary_pos_emb_partial(query_states, key_states, cos, sin, ndim=cos.shape[-1])


class MiniMaxM2Layer(DecoderOnlyLayer):
    """
    MiniMax-M2 decoder layer wrapper.

    Upstream names:
    - `input_layernorm`
    - `self_attn`
    - `post_attention_layernorm`
    - `block_sparse_moe`
    """

    def __init__(
        self,
        layer,
        self_attn: DecoderOnlyAttention,
        lora_config: Optional[RBLNLoRAConfig] = None,
    ):
        nn.Module.__init__(self)
        self.pre_attention_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.pre_feedforward_layernorm = None
        self.post_feedforward_layernorm = None

        self.self_attn = self_attn
        self._phase = "prefill"
        self.lora_config = lora_config

        # Wrap MoE block into an RBLN-friendly module.
        self.mlp = MiniMaxM2SparseMoeBlock(layer.block_sparse_moe)

    def get_mlp(self) -> nn.Module:
        return self.mlp


class MiniMaxM2SparseMoeBlock(nn.Module):
    """
    MiniMax-M2 MoE block wrapper.

    This implementation uses the RBLN custom MoE GLU op to avoid Python-level expert loops.

    Important:
    - MiniMax routing uses sigmoid-based scores and sum-normalization.
    - The RBLN `custom_moe_glu` op takes `router_logits` and a `norm_topk_prob` flag.
      We pass `norm_topk_prob=False` to indicate non-softmax normalization.
    """

    def __init__(self, moe_block: nn.Module):
        super().__init__()
        self.top_k = moe_block.top_k
        self.gate = moe_block.gate
        self.experts = moe_block.experts
        self.num_experts = len(self.experts)

        # Stack expert weights for the fused MoE op.
        # MiniMax expert MLP uses SwiGLU-style: silu(w1(x)) * w3(x) -> w2(...)
        w1 = torch.stack([e.w1.weight for e in self.experts], dim=0)
        w3 = torch.stack([e.w3.weight for e in self.experts], dim=0)
        w2 = torch.stack([e.w2.weight for e in self.experts], dim=0)

        w1_scale = torch.stack(
            [e.w1.weight_scale_inv.repeat_interleave(e.w1.block_size[0], 0) for e in self.experts], dim=0
        )
        w3_scale = torch.stack(
            [e.w3.weight_scale_inv.repeat_interleave(e.w3.block_size[0], 0) for e in self.experts], dim=0
        )
        w2_scale = torch.stack(
            [e.w2.weight_scale_inv.repeat_interleave(e.w2.block_size[0], 0) for e in self.experts], dim=0
        )

        self.block_size = self.experts[0].w1.block_size
        # Register as parameters for tracing/compilation.
        self.w1 = nn.Parameter(w1, requires_grad=False)
        self.w3 = nn.Parameter(w3, requires_grad=False)
        self.w2 = nn.Parameter(w2, requires_grad=False)
        self.w1_scale = nn.Parameter(w1_scale, requires_grad=False)
        self.w3_scale = nn.Parameter(w3_scale, requires_grad=False)
        self.w2_scale = nn.Parameter(w2_scale, requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)

        # router_logits: [B*T, num_experts]
        router_logits = self.gate(x)
        y = torch.ops.rbln_custom_ops.custom_moe_swiglu_fp8(
            x,
            self.w1,
            self.w1_scale,
            self.w3,
            self.w3_scale,
            self.w2,
            self.w2_scale,
            router_logits,
            int(self.top_k),
            False,  # norm_topk_prob: False => non-softmax normalization (matches MiniMax's sum-normalization better)
        )
        return y.view(batch_size, seq_len, hidden_dim)
