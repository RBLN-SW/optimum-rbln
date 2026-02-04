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

from pydantic import Field, field_validator

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger


logger = get_logger(__name__)


class RBLNAutoencoderKLCosmosConfig(RBLNModelConfig):
    """Configuration class for RBLN Cosmos Variational Autoencoder (VAE) models."""

    batch_size: int = Field(default=1, description="The batch size for inference.")
    uses_encoder: bool | None = Field(
        default=None,
        description="Whether to include the encoder part of the VAE in the model. "
        "When False, only the decoder is used (for latent-to-video conversion).",
    )
    num_frames: int = Field(default=121, description="The number of frames in the generated video.")
    height: int = Field(default=704, description="The height in pixels of the generated video.")
    width: int = Field(default=1280, description="The width in pixels of the generated video.")
    num_channels_latents: int | None = Field(default=None, description="The number of channels in latent space.")
    vae_scale_factor_temporal: int | None = Field(
        default=None,
        description="The scaling factor between time space and latent space. "
        "Determines how much shorter the latent representations are compared to the original videos.",
    )
    vae_scale_factor_spatial: int | None = Field(
        default=None,
        description="The scaling factor between pixel space and latent space. "
        "Determines how much smaller the latent representations are compared to the original videos.",
    )
    use_slicing: bool = Field(
        default=False,
        description="Enable sliced VAE encoding and decoding. "
        "If True, the VAE will split the input tensor in slices to compute encoding or decoding in several steps.",
    )

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        if v is None:
            return 1
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")
        if v > 1:
            logger.warning("The batch size of Cosmos VAE Decoder will be explicitly 1 for memory efficiency.")
            return 1
        return v

    @field_validator("num_frames", mode="before")
    @classmethod
    def validate_num_frames(cls, v: int | None) -> int:
        return v if v is not None else 121

    @field_validator("height", mode="before")
    @classmethod
    def validate_height(cls, v: int | None) -> int:
        return v if v is not None else 704

    @field_validator("width", mode="before")
    @classmethod
    def validate_width(cls, v: int | None) -> int:
        return v if v is not None else 1280

    @field_validator("use_slicing", mode="before")
    @classmethod
    def validate_use_slicing(cls, v: bool | None) -> bool:
        return v if v is not None else False

    @property
    def image_size(self):
        return (self.height, self.width)
