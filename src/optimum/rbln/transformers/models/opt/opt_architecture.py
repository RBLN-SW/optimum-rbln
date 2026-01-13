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

from typing import TYPE_CHECKING

import torch.nn as nn

from ...models.decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
)


if TYPE_CHECKING:
    from transformers import OPTForCausalLM


class OPTWrapper(DecoderOnlyWrapper):
    _use_learned_pos_emb = True

    def get_rbln_attn_class(self):
        return OPTAttention

    def get_rbln_layer_class(self):
        return OPTDecoderLayer

    def get_rbln_model_class(self):
        return OPTModel

    def get_model_layer(self, model: "OPTForCausalLM"):
        return model.model.decoder if self.is_causal_lm else model.decoder

    def get_decoder_layers(self, model: "OPTForCausalLM"):
        return model.model.decoder.layers if self.is_causal_lm else model.decoder.layers


class OPTAttention(DecoderOnlyAttention):
    pass


class OPTModel(DecoderOnlyModel):
    def get_embedding(self) -> nn.Embedding:
        return self.embed_tokens

    def get_pos_embedding(self):
        return self.embed_positions

    def get_last_layernorm(self) -> nn.LayerNorm:
        return self.norm


class OPTDecoderLayer(DecoderOnlyLayer):
    pass
