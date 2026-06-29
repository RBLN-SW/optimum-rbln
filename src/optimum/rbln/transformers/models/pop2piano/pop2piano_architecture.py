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

from typing import Tuple

import torch

from ..t5.t5_architecture import T5DecoderWrapper, T5EncoderWrapper


class Pop2PianoWrapper:
    def __init__(self, model, enc_max_seq_len: int, dec_max_seq_len: int = None):
        self.encoder = Pop2PianoEncoderWrapper(model, enc_max_seq_len)
        self.decoder = T5DecoderWrapper(model, dec_max_seq_len=dec_max_seq_len)


class Pop2PianoEncoderWrapper(T5EncoderWrapper):
    """Pop2Piano shares T5's encoder blocks and cross-kv projection layout; the
    only difference is that the encoder is fed mel features as `inputs_embeds`
    instead of `input_ids` (the token embedding lookup is bypassed)."""

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        b_idx: torch.Tensor,
        *cross_key_values: Tuple[torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        # Pass the 2D mask straight through; the encoder stack builds the 4D
        # bidirectional mask itself (transformers v5 `create_bidirectional_mask`).
        encoder_outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_hidden_states = encoder_outputs[0]

        cross_kv = []
        for k_proj, v_proj in zip(self.cross_k_projects, self.cross_v_projects, strict=False):
            past_k = (
                k_proj(last_hidden_states).view(1, self.encoder_max_length, self.num_heads, self.d_kv).transpose(1, 2)
            )
            past_v = (
                v_proj(last_hidden_states).view(1, self.encoder_max_length, self.num_heads, self.d_kv).transpose(1, 2)
            )
            cross_kv.append(past_k)
            cross_kv.append(past_v)

        batch_axis = torch.tensor(0, dtype=torch.int16)
        cross_key_values = list(cross_key_values)
        for i in range(self.n_layer * 2):
            cross_key_values[i] = torch.ops.rbln_custom_ops.rbln_cache_update(
                cross_key_values[i], cross_kv[i], b_idx[0], batch_axis
            )

        return cross_key_values
