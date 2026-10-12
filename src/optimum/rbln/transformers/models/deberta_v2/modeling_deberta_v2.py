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


class DebertaV2ModelMixin:
    rbln_model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    @classmethod
    def _update_rbln_config(cls, preprocessors, model, model_config, rbln_config):
        if model_config.type_vocab_size == 0:
            names = rbln_config.model_input_names or cls.rbln_model_input_names
            rbln_config.model_input_names = [name for name in names if name != "token_type_ids"]
        return super()._update_rbln_config(preprocessors, model, model_config, rbln_config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_dict=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        names = self.rbln_config.model_input_names or self.rbln_model_input_names
        return super().forward(*(inputs[name] for name in names), return_dict=return_dict, **kwargs)


class RBLNDebertaV2Model(DebertaV2ModelMixin, RBLNTransformerEncoderForFeatureExtraction):
    """DeBERTa-v2 and DeBERTa-v3 encoder for contextual token embeddings."""


class RBLNDebertaV2ForMaskedLM(DebertaV2ModelMixin, RBLNModelForMaskedLM):
    """DeBERTa-v2 and DeBERTa-v3 masked language model."""


class RBLNDebertaV2ForSequenceClassification(DebertaV2ModelMixin, RBLNModelForSequenceClassification):
    """DeBERTa-v2 and DeBERTa-v3 sequence classifier."""


class RBLNDebertaV2ForQuestionAnswering(DebertaV2ModelMixin, RBLNModelForQuestionAnswering):
    """DeBERTa-v2 and DeBERTa-v3 extractive question answering model."""


__all__ = [
    "RBLNDebertaV2Model",
    "RBLNDebertaV2ForMaskedLM",
    "RBLNDebertaV2ForSequenceClassification",
    "RBLNDebertaV2ForQuestionAnswering",
]
