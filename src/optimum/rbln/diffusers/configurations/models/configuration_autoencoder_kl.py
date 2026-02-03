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

from pydantic import field_validator

from ....configuration_utils import RBLNModelConfig


class RBLNAutoencoderKLConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Variational Autoencoder (VAE) models.

    This class inherits from RBLNModelConfig and provides specific configuration options
    for VAE models used in diffusion-based image generation.
    """

    batch_size: int = 1
    sample_size: tuple[int, int] | int | None = None
    uses_encoder: bool | None = None
    vae_scale_factor: float | None = None
    in_channels: int | None = None
    latent_channels: int | None = None

    def __init__(self, **data: Any):
        # Normalize sample_size to tuple
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
    def image_size(self) -> tuple[int, int] | None:
        return self.sample_size

    @property
    def latent_sample_size(self) -> tuple[int, int] | None:
        if self.image_size is None or self.vae_scale_factor is None:
            return None
        return (int(self.image_size[0] // self.vae_scale_factor), int(self.image_size[1] // self.vae_scale_factor))
