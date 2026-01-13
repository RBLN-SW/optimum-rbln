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
from typing import TYPE_CHECKING, Tuple, Union

import torch
import torch.nn as nn

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
)


if TYPE_CHECKING:
    from transformers import GPT2LMHeadModel, GPT2Model


class GPT2Wrapper(DecoderOnlyWrapper):
    def get_rbln_attn_class(self):
        return GPT2Attention

    def get_rbln_layer_class(self):
        return GPT2Layer

    def get_rbln_model_class(self):
        return GPT2Model

    def get_attn_layer(self, layer: nn.Module):
        return layer.attn

    def get_model_layer(self, model: Union["GPT2LMHeadModel", "GPT2Model"]):
        return model.transformer if self.is_causal_lm else model

    def get_decoder_layers(self, model: Union["GPT2LMHeadModel", "GPT2Model"]):
        return model.transformer.h if self.is_causal_lm else model.h


class GPT2Model(DecoderOnlyModel):
    def get_last_layernorm(self) -> nn.LayerNorm:
        return self.norm

    def get_embedding(self) -> nn.Embedding:
        return self.embed_tokens

    def get_pos_embedding(self) -> nn.Embedding:
        return self.embed_positions


class GPT2Layer(DecoderOnlyLayer):
    pass


class GPT2Attention(DecoderOnlyAttention):
    # These attributes are accessed during DecoderOnlyAttention.__init__() via get_attn_scale().
    # Define safe defaults at class level so they exist before __post_init__ runs.
    scale_attn_weights = True
    scale_attn_by_inverse_layer_idx = False

    def __post_init__(self, self_attn):
        self.c_attn = self_attn.c_attn
        self.o_proj = self_attn.c_proj
        self.split_size = self_attn.split_size
        self.scale_attn_weights = getattr(self_attn, "scale_attn_weights", True)
        self.scale_attn_by_inverse_layer_idx = getattr(self_attn, "scale_attn_by_inverse_layer_idx", False)

    def projection(self, hidden_states, lora_int_id) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if lora_int_id is not None:
            raise NotImplementedError("LoRA is not supported for GPT2Attention")

        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        return query_states, key_states, value_states

    def get_attn_scale(self):
        scale = 1.0
        if self.scale_attn_weights:
            scale /= math.sqrt(self.head_dim)

        if self.scale_attn_by_inverse_layer_idx:
            scale /= 1 + self.layer_idx

        return scale
