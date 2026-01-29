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

from ...models.decoderonly.decoderonly_architecture import DecoderOnlyWrapper


if TYPE_CHECKING:
    from transformers import OPTForCausalLM


class OPTWrapper(DecoderOnlyWrapper):
    _use_learned_pos_emb = True

    def get_model_layer(self, model: "OPTForCausalLM"):
        return model.model.decoder if self.is_causal_lm else model.decoder

    def get_decoder_layers(self, model: "OPTForCausalLM"):
        return model.model.decoder.layers if self.is_causal_lm else model.decoder.layers
