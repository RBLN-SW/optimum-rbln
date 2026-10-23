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
from transformers.image_utils import SizeDict

from ...modeling_generic import RBLNImageModel, RBLNModelForImageClassification


class Dinov2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False,
        )


class Dinov2Mixin:
    @classmethod
    def _wrap_model_if_needed(cls, model, rbln_config):
        compiled_model = deepcopy(model)
        compiled_model.set_attn_implementation("eager")
        return Dinov2Wrapper(compiled_model).eval()

    @classmethod
    def _update_rbln_config(cls, preprocessors, model, model_config, rbln_config):
        if rbln_config.image_size is None:
            for processor in preprocessors or []:
                crop_size = getattr(processor, "crop_size", None)
                if getattr(processor, "do_center_crop", False) and isinstance(crop_size, (dict, SizeDict)):
                    if "height" in crop_size and "width" in crop_size:
                        rbln_config.image_size = (crop_size["height"], crop_size["width"])
                        break
        return super()._update_rbln_config(preprocessors, model, model_config, rbln_config)


class RBLNDinov2Model(Dinov2Mixin, RBLNImageModel):
    pass


class RBLNDinov2ForImageClassification(Dinov2Mixin, RBLNModelForImageClassification):
    pass


__all__ = ["RBLNDinov2ForImageClassification", "RBLNDinov2Model"]
