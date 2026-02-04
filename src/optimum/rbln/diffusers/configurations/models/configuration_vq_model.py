# Copyright 2025 Rebellions Inc. All rights reserved.

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

from typing import Any

from pydantic import Field, field_validator

from ....configuration_utils import RBLNModelConfig


class RBLNVQModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN VQModel models, used in Kandinsky.

    This class inherits from RBLNModelConfig and provides specific configuration options
    for VQModel, which acts similarly to a VAE but uses vector quantization.
    """

    batch_size: int = Field(default=1, description="The batch size for inference.")
    sample_size: tuple[int, int] | None = Field(
        default=None,
        description="The spatial dimensions (height, width) of the input/output images. "
        "If an integer is provided, it's used for both height and width.",
    )
    uses_encoder: bool | None = Field(
        default=None,
        description="Whether to include the encoder part of the VAE in the model. "
        "When False, only the decoder is used (for latent-to-image conversion).",
    )
    vqmodel_scale_factor: float | None = Field(
        default=None,
        description="The scaling factor between pixel space and latent space. "
        "Determines the downsampling ratio between original images and latent representations.",
    )
    in_channels: int | None = Field(default=None, description="Number of input channels for the model.")
    latent_channels: int | None = Field(default=None, description="Number of channels in the latent space.")

    def __init__(self, **data: Any):
        # Normalize sample_size before super().__init__
        sample_size = data.get("sample_size")
        if isinstance(sample_size, int):
            data["sample_size"] = (sample_size, sample_size)

        super().__init__(**data)

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        if v is None:
            return 1
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")
        return v

    @property
    def image_size(self):
        return self.sample_size

    @property
    def latent_sample_size(self):
        return (
            int(self.image_size[0] // self.vqmodel_scale_factor),
            int(self.image_size[1] // self.vqmodel_scale_factor),
        )
