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

from ..decoderonly.decoderonly_architecture import DecoderOnlyAttention, DecoderOnlyWrapper


class SmolLM3Wrapper(DecoderOnlyWrapper):
    def get_rbln_attn_class(self):
        return SmolLM3Attention


class SmolLM3Attention(DecoderOnlyAttention):
    def __post_init__(self, self_attn):
        super().__post_init__(self_attn)
        self.use_rope = self_attn.use_rope

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        if not self.use_rope:
            return query_states, key_states
        return super().apply_rotary_pos_embed(query_states, key_states, cos, sin)
