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
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndNoAttention, ImageClassifierOutput

from ...modeling_generic import RBLNImageModel, RBLNModelForImageClassification


class ConvNextWrapper(torch.nn.Module):
    def __init__(self, model, classifier):
        super().__init__()
        self.model = model
        self.classifier = classifier

    def forward(self, pixel_values):
        output = self.model(pixel_values=pixel_values, output_hidden_states=False, return_dict=False)
        return output[0] if self.classifier else output[:2]


class ConvNextMixin:
    _classifier = False

    @classmethod
    def _wrap_model_if_needed(cls, model, rbln_config):
        return ConvNextWrapper(model, cls._classifier).eval()


class RBLNConvNextModel(ConvNextMixin, RBLNImageModel):
    """ConvNeXt spatial feature map and pooled embedding."""

    _output_class = BaseModelOutputWithPoolingAndNoAttention


class RBLNConvNextForImageClassification(ConvNextMixin, RBLNModelForImageClassification):
    """ConvNeXt image classification logits."""

    _classifier = True
    _output_class = ImageClassifierOutput


__all__ = ["RBLNConvNextModel", "RBLNConvNextForImageClassification"]
