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

from torch import nn
from transformers import AutoModelForPreTraining

from ...modeling_generic import (
    RBLNModelForSequenceClassification,
    RBLNTransformerEncoder,
    RBLNTransformerEncoderForFeatureExtraction,
)


class ElectraModelWrapper(nn.Module):
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


class ElectraModelMixin:
    rbln_model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    @classmethod
    def _wrap_model_if_needed(cls, model, rbln_config):
        model = deepcopy(model)
        model.set_attn_implementation("eager")
        return ElectraModelWrapper(model).eval()


class RBLNElectraModel(ElectraModelMixin, RBLNTransformerEncoderForFeatureExtraction):
    pass


class RBLNElectraForSequenceClassification(ElectraModelMixin, RBLNModelForSequenceClassification):
    pass


class RBLNElectraForPreTraining(ElectraModelMixin, RBLNTransformerEncoder):
    auto_model_class = AutoModelForPreTraining


__all__ = ["RBLNElectraForPreTraining", "RBLNElectraForSequenceClassification", "RBLNElectraModel"]
