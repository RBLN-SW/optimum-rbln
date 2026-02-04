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

from pydantic import Field, PrivateAttr, field_validator

from ....configuration_utils import RBLNModelConfig


class RBLNControlNetModelConfig(RBLNModelConfig):
    """Configuration class for RBLN ControlNet models."""

    batch_size: int = Field(default=1, description="The batch size for inference.")
    max_seq_len: int | None = Field(
        default=None,
        description="Maximum sequence length for text inputs when used with cross-attention.",
    )
    unet_sample_size: tuple[int, int] | None = Field(
        default=None, description="The spatial dimensions (height, width) of UNet samples."
    )
    vae_sample_size: tuple[int, int] | None = Field(
        default=None, description="The spatial dimensions (height, width) of VAE samples."
    )
    text_model_hidden_size: int | None = Field(default=None, description="Hidden size of the text encoder model.")

    _batch_size_is_specified: bool = PrivateAttr(default=False)

    def __init__(self, **data: Any):
        batch_size_specified = "batch_size" in data and data["batch_size"] is not None
        super().__init__(**data)
        self._batch_size_is_specified = batch_size_specified

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        if v is None:
            return 1
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")
        return v

    @property
    def batch_size_is_specified(self) -> bool:
        return self._batch_size_is_specified
