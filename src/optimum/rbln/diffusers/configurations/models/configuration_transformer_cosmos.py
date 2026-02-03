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


class RBLNCosmosTransformer3DModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Cosmos Transformer models.

    This class inherits from RBLNModelConfig and provides specific configuration options
    for Transformer models used in diffusion models like Cosmos.
    """

    batch_size: int = 1
    num_frames: int = 121
    height: int = 704
    width: int = 1280
    fps: int = 30
    max_seq_len: int | None = None
    embedding_dim: int | None = None
    num_channels_latents: int | None = None
    num_latent_frames: int | None = None
    latent_height: int | None = None
    latent_width: int | None = None

    def __init__(self, **data: Any):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            num_frames (Optional[int]): The number of frames in the generated video. Defaults to 121.
            height (Optional[int]): The height in pixels of the generated video. Defaults to 704.
            width (Optional[int]): The width in pixels of the generated video. Defaults to 1280.
            fps (Optional[int]): The frames per second of the generated video.  Defaults to 30.
            max_seq_len (Optional[int]): Maximum sequence length of prompt embeds.
            embedding_dim (Optional[int]): Embedding vector dimension of prompt embeds.
            num_channels_latents (Optional[int]): The number of channels in latent space.
            latent_height (Optional[int]): The height in pixels in latent space.
            latent_width (Optional[int]): The width in pixels in latent space.
            **data: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
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
