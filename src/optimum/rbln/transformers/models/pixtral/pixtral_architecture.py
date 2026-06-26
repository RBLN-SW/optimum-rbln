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

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..decoderonly.decoderonly_architecture import apply_rotary_pos_emb


class PixtralAttention(nn.Module):
    def __init__(self, self_attention):
        super().__init__()
        self.original_model = self_attention
        self.num_heads = (
            getattr(self.original_model, "num_heads", None) or self.original_model.config.num_attention_heads
        )
        self.head_dim = self.original_model.head_dim
        self.scaling = self.head_dim**-0.5

        self.__post_init__()

    def __post_init__(self):
        self.q_proj = self.original_model.q_proj
        self.k_proj = self.original_model.k_proj
        self.v_proj = self.original_model.v_proj
        self.o_proj = self.original_model.o_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ):
        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # TODO: return output attention
        query_states = query_states.view(batch_size, patches, 1, self.num_heads, self.head_dim).transpose(1, 3)
        key_states = key_states.view(batch_size, patches, 1, self.num_heads, self.head_dim).transpose(1, 3)
        value_states = value_states.view(batch_size, patches, 1, self.num_heads, self.head_dim).transpose(1, 3)

        cos, sin = position_embeddings
        cos = cos[None, None, None, :, :]
        sin = sin[None, None, None, :, :]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_weights = torch.matmul(query_states, key_states.transpose(3, 4)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 3)

        attn_output = attn_output.reshape(batch_size, patches, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, _
