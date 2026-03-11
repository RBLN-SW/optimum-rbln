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

from ...models.decoderonly.decoderonly_architecture import DecoderOnlyWrapper, DecoderOnlyForCausalLM, DecoderOnlyModel, DecoderOnlyLayer, DecoderOnlyAttention   


class AXK1Wrapper(DecoderOnlyWrapper):
    def get_rbln_attn_class(self):
        return AXK1Attention

    def get_rbln_layer_class(self):
        return AXK1Layer

    def get_rbln_model_class(self):
        return AXK1Model
        
    def get_rbln_causal_lm_class(self):
        return AXK1ForCausalLM

class AXK1ForCausalLM(DecoderOnlyForCausalLM):
    pass

class AXK1Model(DecoderOnlyModel):
    pass        

class AXK1Layer(DecoderOnlyLayer):
    pass

class AXK1Attention(DecoderOnlyAttention):
    pass