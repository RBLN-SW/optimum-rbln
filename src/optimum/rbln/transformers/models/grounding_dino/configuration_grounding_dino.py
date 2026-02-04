# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, ClassVar

import torch
from pydantic import Field

from ...configuration_generic import RBLNImageModelConfig, RBLNModelConfig


class RBLNGroundingDinoForObjectDetectionConfig(RBLNImageModelConfig):
    """Configuration for RBLN Grounding DINO for object detection."""

    submodules: ClassVar[list[str]] = [
        "text_backbone",
        "backbone",
        "encoder",
        "decoder",
    ]

    encoder: RBLNModelConfig | None = Field(default=None, description="Configuration for the encoder component.")
    decoder: RBLNModelConfig | None = Field(default=None, description="Configuration for the decoder component.")
    text_backbone: RBLNModelConfig | None = Field(
        default=None, description="Configuration for the text backbone component."
    )
    backbone: RBLNModelConfig | None = Field(default=None, description="Configuration for the backbone component.")
    output_attentions: bool = Field(default=False, description="Whether to output attentions.")
    output_hidden_states: bool = Field(default=False, description="Whether to output hidden states.")

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Initialize submodules if not already converted by model_validator
        if self.encoder is None or isinstance(self.encoder, dict):
            self.encoder = self.initialize_submodule_config(submodule_config=self.encoder, batch_size=self.batch_size)
        if self.decoder is None or isinstance(self.decoder, dict):
            self.decoder = self.initialize_submodule_config(submodule_config=self.decoder, batch_size=self.batch_size)
        if self.text_backbone is None or isinstance(self.text_backbone, dict):
            self.text_backbone = self.initialize_submodule_config(
                submodule_config=self.text_backbone, batch_size=self.batch_size
            )
        if self.backbone is None or isinstance(self.backbone, dict):
            self.backbone = self.initialize_submodule_config(
                submodule_config=self.backbone, batch_size=self.batch_size
            )


class RBLNGroundingDinoComponentConfig(RBLNImageModelConfig):
    """Configuration for RBLN Grounding DINO component."""

    spatial_shapes_list: list[tuple[int, int]] | None = Field(
        default=None, description="List of spatial shapes for the component."
    )
    output_attentions: bool = Field(default=False, description="Whether to output attentions.")
    output_hidden_states: bool = Field(default=False, description="Whether to output hidden states.")

    @property
    def spatial_shapes(self):
        if self.spatial_shapes_list is None:
            raise ValueError("Spatial shapes are not defined. Please set them before accessing.")
        return torch.tensor(self.spatial_shapes_list)


class RBLNGroundingDinoEncoderConfig(RBLNGroundingDinoComponentConfig):
    pass


class RBLNGroundingDinoDecoderConfig(RBLNGroundingDinoComponentConfig):
    pass
