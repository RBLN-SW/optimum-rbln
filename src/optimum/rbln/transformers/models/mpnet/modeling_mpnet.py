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

from ...modeling_generic import RBLNTransformerEncoderForFeatureExtraction
from .mpnet_architecture import MPNetModelWrapper


class RBLNMPNetModel(RBLNTransformerEncoderForFeatureExtraction):
    rbln_model_input_names = ["input_ids", "attention_mask"]

    def forward(self, *args, output_hidden_states=None, output_attentions=None, **kwargs):
        if output_hidden_states or (output_hidden_states is None and self.config.output_hidden_states):
            raise NotImplementedError("MPNet does not support `output_hidden_states=True`.")
        if output_attentions or (output_attentions is None and self.config.output_attentions):
            raise NotImplementedError("MPNet does not support `output_attentions=True`.")
        return super().forward(*args, **kwargs)

    def _prepare_output(self, output, return_dict):
        output = (output,) if not isinstance(output, (tuple, list)) else tuple(output)
        if len(output) == 1:
            output += (None,)
        return super()._prepare_output(output, return_dict)

    @classmethod
    def _wrap_model_if_needed(cls, model, rbln_config):
        return MPNetModelWrapper(model).eval()

    @classmethod
    def _update_rbln_config(cls, preprocessors, model, model_config, rbln_config):
        max_seq_len = model_config.max_position_embeddings - model.embeddings.padding_idx - 1
        if rbln_config.max_seq_len is None:
            rbln_config.max_seq_len = max_seq_len
        lengths = [rbln_config.max_seq_len] if isinstance(rbln_config.max_seq_len, int) else rbln_config.max_seq_len
        if max(lengths) > max_seq_len:
            raise ValueError(f"`max_seq_len` must not exceed MPNet's position limit ({max_seq_len}).")
        return super()._update_rbln_config(preprocessors, model, model_config, rbln_config)


__all__ = ["RBLNMPNetModel"]
