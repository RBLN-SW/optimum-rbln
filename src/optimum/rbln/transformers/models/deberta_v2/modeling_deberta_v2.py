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

from copy import deepcopy

import torch
from torch import nn

from ...modeling_generic import (
    RBLNModelForSequenceClassification,
    RBLNTransformerEncoderForFeatureExtraction,
)


class DebertaV2ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False,
        )


class DebertaV2ModelMixin:
    @classmethod
    def _wrap_model_if_needed(cls, model, rbln_config):
        return DebertaV2ModelWrapper(deepcopy(model)).eval()

    @classmethod
    def _update_rbln_config(cls, preprocessors, model, model_config, rbln_config):
        if rbln_config.model_input_names is None:
            rbln_config.model_input_names = ["input_ids", "attention_mask"]
            if model_config.type_vocab_size > 0:
                rbln_config.model_input_names.append("token_type_ids")
        elif model_config.type_vocab_size == 0:
            # The v3 tokenizer advertises token types even when the model has
            # no type embeddings. An unused traced input cannot be compiled.
            rbln_config.model_input_names = [
                name for name in rbln_config.model_input_names if name != "token_type_ids"
            ]
        return super()._update_rbln_config(preprocessors, model, model_config, rbln_config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_dict=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        names = self.rbln_config.model_input_names
        return super().forward(*(inputs[name] for name in names), return_dict=return_dict, **kwargs)


class RBLNDebertaV2Model(DebertaV2ModelMixin, RBLNTransformerEncoderForFeatureExtraction):
    pass


class RBLNDebertaV2ForSequenceClassification(DebertaV2ModelMixin, RBLNModelForSequenceClassification):
    pass


__all__ = ["RBLNDebertaV2ForSequenceClassification", "RBLNDebertaV2Model"]
