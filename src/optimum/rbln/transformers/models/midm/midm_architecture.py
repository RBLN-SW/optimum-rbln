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

import math
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
    apply_rotary_pos_emb_partial,
    rotate_half,
)


if TYPE_CHECKING:
    from transformers import PreTrainedModel as MidmLMHeadModel

    from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig


def apply_rotary_to_tensor(tensor, cos, sin, rot_dim):
    """Applies rotary position embedding to the specified dimension of the tensor."""
    tensor_, tensor_pass = tensor[..., :rot_dim], tensor[..., rot_dim:]
    tensor_embed = (tensor_ * cos) + (rotate_half(tensor_) * sin)
    return torch.cat((tensor_embed, tensor_pass), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors."""
    rot_dim = cos.shape[-1]
    q_embed = apply_rotary_to_tensor(q, cos, sin, rot_dim)
    k_embed = apply_rotary_to_tensor(k, cos, sin, rot_dim)
    return q_embed, k_embed


class MidmLMHeadModelWrapper(DecoderOnlyWrapper):
    def get_rotary_emb(self, max_seq_len):
        self.config.rope_theta = 10000
        self.config.head_dim = self.config.n_embd // self.config.n_head
        self.config.partial_rotary_factor = self.config.rotary_percentage
        return super().get_rotary_emb(max_seq_len=max_seq_len)

    def get_rbln_attn_class(self):
        return MidmAttention

    def get_rbln_layer_class(self):
        return MidmLayer

    def get_rbln_model_class(self):
        return MidmModel

    def get_model_layer(self, causal_lm: "MidmLMHeadModel"):
        return causal_lm.transformer

    def get_decoder_layers(self, causal_lm: "MidmLMHeadModel"):
        return causal_lm.transformer.h


class MidmModel(DecoderOnlyModel):
    def __init__(self, model, layers, rbln_config, use_learned_pos_emb=None):
        super().__init__(model, layers, rbln_config, use_learned_pos_emb=use_learned_pos_emb)
        self.use_layernorm1p = getattr(model, "use_layernorm1p", False)

    def get_layernorm1p(self, module: nn.LayerNorm):
        def layernorm1p(input: torch.Tensor):
            """Applies Layer Normalization with a slight modification on the weights."""
            return torch.nn.functional.layer_norm(
                input, module.normalized_shape, module.weight + 1, module.bias, module.eps
            )

        return layernorm1p

    def get_last_layernorm(self) -> nn.LayerNorm:
        if self.use_layernorm1p:
            return self.get_layernorm1p(self.norm)
        return self.norm

    def get_embedding(self) -> nn.Embedding:
        return self.embed_tokens

    def get_pos_embedding(self) -> nn.Embedding:
        return self.embed_positions


class MidmLayer(DecoderOnlyLayer):
    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config=None):
        super().__init__(layer, self_attn, lora_config)
        self.use_layernorm1p = getattr(layer, "use_layernorm1p", False)

    def get_layernorm1p(self, module: nn.LayerNorm):
        def layernorm1p(input: torch.Tensor):
            """Applies Layer Normalization with a slight modification on the weights."""
            return torch.nn.functional.layer_norm(
                input, module.normalized_shape, module.weight + 1, module.bias, module.eps
            )

        return layernorm1p

    def get_pre_attention_layernorm(self) -> nn.LayerNorm:
        if self.use_layernorm1p:
            return self.get_layernorm1p(self.pre_attention_layernorm)
        return self.pre_attention_layernorm

    def get_post_attention_layernorm(self) -> nn.LayerNorm:
        if self.use_layernorm1p:
            return self.get_layernorm1p(self.post_attention_layernorm)
        return self.post_attention_layernorm


class MidmAttention(DecoderOnlyAttention):
    def __init__(
        self,
        self_attn,
        rbln_config: "RBLNDecoderOnlyModelConfig",
        is_sliding=False,
    ):
        super().__init__(self_attn, rbln_config, is_sliding)
        self.c_attn = self_attn.c_attn
        self.o_proj = self_attn.c_proj
        self.split_size = self_attn.split_size
        self.num_key_value_heads = self_attn.num_heads
        self.scale_attn_weights = getattr(self_attn, "scale_attn_weights", True)
        self.scale_attn_by_inverse_layer_idx = getattr(self_attn, "scale_attn_by_inverse_layer_idx", False)
        self.scale_qk_by_inverse_layer_idx = getattr(self_attn, "scale_qk_by_inverse_layer_idx", False)

    def projection(self, hidden_states, lora_int_id) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if lora_int_id is not None:
            raise NotImplementedError("LoRA is not supported for MidmAttention")

        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        return query_states, key_states, value_states

    def get_attn_scale(self):
        scale = 1.0
        if self.scale_attn_weights:
            scale /= math.sqrt(self.head_dim)

        if self.scale_attn_by_inverse_layer_idx and not self.scale_qk_by_inverse_layer_idx:
            scale /= 1 + self.layer_idx

        return scale

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        return apply_rotary_pos_emb_partial(query_states, key_states, cos, sin, ndim=cos.shape[-1])
