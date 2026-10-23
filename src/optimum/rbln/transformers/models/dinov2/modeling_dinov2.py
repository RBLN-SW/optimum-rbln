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
from transformers.image_utils import SizeDict

from ...modeling_generic import RBLNImageModel, RBLNModelForImageClassification


class Dinov2EmbeddingsWrapper(nn.Module):
    def __init__(self, embeddings, image_size):
        super().__init__()
        self.embeddings = embeddings
        height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
        patches = (height // embeddings.patch_size) * (width // embeddings.patch_size)
        dummy = embeddings.position_embeddings.new_empty(1, patches + 1, embeddings.position_embeddings.shape[-1])
        with torch.no_grad():
            position = embeddings.interpolate_pos_encoding(dummy, height, width).detach().clone()
        self.register_buffer("position", position)

    def forward(self, pixel_values, bool_masked_pos=None):
        source = self.embeddings
        embeddings = source.patch_embeddings(pixel_values.to(dtype=source.patch_embeddings.projection.weight.dtype))
        if bool_masked_pos is not None and source.use_mask_token:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), source.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )
        cls_tokens = source.cls_token.expand(pixel_values.shape[0], -1, -1)
        return source.dropout(torch.cat((cls_tokens, embeddings), dim=1) + self.position)


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
        model = deepcopy(model)
        backbone = getattr(model, "dinov2", model)
        backbone.embeddings = Dinov2EmbeddingsWrapper(
            backbone.embeddings, (rbln_config.image_height, rbln_config.image_width)
        )
        return Dinov2Wrapper(model).eval()

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
