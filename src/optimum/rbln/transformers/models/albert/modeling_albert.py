# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from ...modeling_generic import (
    RBLNModelForMaskedLM,
    RBLNModelForQuestionAnswering,
    RBLNModelForSequenceClassification,
    RBLNTransformerEncoderForFeatureExtraction,
)


class AlbertModelMixin:
    rbln_model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    @classmethod
    def _update_rbln_config(cls, preprocessors, model, model_config, rbln_config):
        # Some tokenizers omit segment IDs from model_input_names even though ALBERT uses them.
        if rbln_config.model_input_names is None:
            rbln_config.model_input_names = cls.rbln_model_input_names
        return super()._update_rbln_config(preprocessors, model, model_config, rbln_config)

    @classmethod
    def _wrap_model_if_needed(cls, model, rbln_config):
        class AlbertWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                return self.model(*args, return_dict=False, **kwargs)

        return AlbertWrapper(model).eval()

    def forward(
        self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, return_dict=None, **kwargs
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        names = self.rbln_config.model_input_names or self.rbln_model_input_names
        if position_ids is not None and "position_ids" not in names:
            raise ValueError("Compile with position_ids in model_input_names to supply custom positions.")
        if "position_ids" in names and position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
        }
        return super().forward(*(inputs[name] for name in names), return_dict=return_dict, **kwargs)


class RBLNAlbertModel(AlbertModelMixin, RBLNTransformerEncoderForFeatureExtraction):
    """ALBERT encoder preserving the model's shared Transformer parameters."""


class RBLNAlbertForMaskedLM(AlbertModelMixin, RBLNModelForMaskedLM):
    """ALBERT masked language model."""


class RBLNAlbertForSequenceClassification(AlbertModelMixin, RBLNModelForSequenceClassification):
    """ALBERT sequence classifier."""


class RBLNAlbertForQuestionAnswering(AlbertModelMixin, RBLNModelForQuestionAnswering):
    """ALBERT extractive question answering model."""


__all__ = [
    "RBLNAlbertModel",
    "RBLNAlbertForMaskedLM",
    "RBLNAlbertForSequenceClassification",
    "RBLNAlbertForQuestionAnswering",
]
