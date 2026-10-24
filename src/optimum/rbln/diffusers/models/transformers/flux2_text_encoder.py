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

import torch
from transformers import Qwen3ForCausalLM
from transformers.modeling_outputs import BaseModelOutput

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ...configurations.models.configuration_flux2_text_encoder import RBLNFlux2KleinTextEncoderConfig


class Flux2KleinTextEncoderWrapper(torch.nn.Module):
    """Preserve every native hidden-state tuple index used by the pipeline."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        ).hidden_states


class RBLNFlux2KleinTextEncoder(RBLNModel):
    """Qwen3 hidden states for FLUX.2 klein's original prompt extraction.

    Every decoder layer is retained. The original pipeline can select hidden
    states at their native indices, including its default layers 9, 18 and 27.
    This component encodes prompts; it does not expose language-model logits
    or generation caches.
    """

    auto_model_class = Qwen3ForCausalLM
    _rbln_config_class = RBLNFlux2KleinTextEncoderConfig
    _output_class = BaseModelOutput

    @classmethod
    def _wrap_model_if_needed(cls, model, rbln_config):
        return Flux2KleinTextEncoderWrapper(model).eval()

    @classmethod
    def _update_rbln_config(cls, preprocessors, model, model_config, rbln_config):
        shape = [rbln_config.batch_size, rbln_config.max_seq_len]
        rbln_config.set_compile_cfgs(
            [RBLNCompileConfig(input_info=[("input_ids", shape, "int64"), ("attention_mask", shape, "int64")])]
        )
        return rbln_config

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True, use_cache=False, return_dict=True):
        if use_cache or not output_hidden_states:
            raise ValueError("This compiled text encoder exposes hidden states without a generation cache")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        hidden = self.model[0](input_ids, attention_mask)
        if not isinstance(hidden, (list, tuple)) or len(hidden) != self.config.num_hidden_layers + 1:
            raise ValueError("Compiled hidden-state tuple does not match the complete native Qwen3 model")
        hidden = tuple(hidden)
        output = BaseModelOutput(last_hidden_state=hidden[-1], hidden_states=hidden)
        return output if return_dict else output.to_tuple()


__all__ = ["RBLNFlux2KleinTextEncoder"]
