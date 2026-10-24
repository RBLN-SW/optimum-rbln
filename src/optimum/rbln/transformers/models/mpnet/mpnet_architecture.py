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

import copy

import torch
from torch import nn
from transformers.models.mpnet.modeling_mpnet import MPNetEncoder


class MPNetSharedPositionBiasEncoder(MPNetEncoder):
    def compute_position_bias(self, hidden_states, position_ids=None, num_buckets=32):
        # Default relative positions are identical across rows. The existing
        # mask add broadcasts this learned bias against each row's mask.
        if position_ids is None:
            hidden_states = hidden_states[:1]
        return super().compute_position_bias(hidden_states, position_ids=position_ids, num_buckets=num_buckets)


class MPNetSDPAAttention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, **kwargs):
        if output_attentions:
            return self.attention(hidden_states, attention_mask, position_bias, output_attentions=True)

        attention = self.attention
        hidden_shape = (*hidden_states.shape[:-1], -1, attention.attention_head_size)
        query = attention.q(hidden_states).view(hidden_shape).transpose(1, 2)
        key = attention.k(hidden_states).view(hidden_shape).transpose(1, 2)
        value = attention.v(hidden_states).view(hidden_shape).transpose(1, 2)
        mask = position_bias
        if attention_mask is not None:
            mask = attention_mask if mask is None else mask + attention_mask
        context = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        context = context.transpose(1, 2).reshape(*hidden_states.shape[:-1], attention.all_head_size)
        return (attention.o(context),)


class MPNetModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.model.encoder.__class__ = MPNetSharedPositionBiasEncoder
        for layer in self.model.encoder.layer:
            layer.attention.attn = MPNetSDPAAttention(layer.attention.attn)

    def forward(self, input_ids, attention_mask=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        if output.pooler_output is None:
            return (output.last_hidden_state,)
        return output.last_hidden_state, output.pooler_output
