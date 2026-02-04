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


class RBLNCosmosTransformer3DModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Cosmos Transformer models.

    This class inherits from RBLNModelConfig and provides specific configuration options
    for Transformer models used in diffusion models like Cosmos.
    """

    batch_size: int = Field(default=1, description="The batch size for inference.")
    num_frames: int = Field(default=121, description="The number of frames in the generated video.")
    height: int = Field(default=704, description="The height in pixels of the generated video.")
    width: int = Field(default=1280, description="The width in pixels of the generated video.")
    fps: int = Field(default=30, description="The frames per second of the generated video.")
    max_seq_len: int | None = Field(default=None, description="Maximum sequence length of prompt embeds.")
    embedding_dim: int | None = Field(default=None, description="Embedding vector dimension of prompt embeds.")
    num_channels_latents: int | None = Field(default=None, description="The number of channels in latent space.")
    num_latent_frames: int | None = Field(default=None, description="The number of frames in latent space.")
    latent_height: int | None = Field(default=None, description="The height in pixels in latent space.")
    latent_width: int | None = Field(default=None, description="The width in pixels in latent space.")

    def __init__(self, **data: Any):
        if data.get("timeout") is None:
            data["timeout"] = 80

        super().__init__(**data)

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        if v is None:
            return 1
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")
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

    @field_validator("fps", mode="before")
    @classmethod
    def validate_fps(cls, v: int | None) -> int:
        return v if v is not None else 30
