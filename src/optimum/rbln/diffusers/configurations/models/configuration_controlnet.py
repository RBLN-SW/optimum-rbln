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

from pydantic import Field, model_validator

from ....configuration_utils import PositiveIntDefaultOne, RBLNModelConfig


class RBLNControlNetModelConfig(RBLNModelConfig):
    """Configuration class for RBLN ControlNet models."""

    batch_size: PositiveIntDefaultOne = Field(default=1, description="The batch size for inference.")
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
    batch_size_is_specified: bool = Field(
        default=False, exclude=True, description="Whether the batch size was explicitly specified by the user."
    )

    @model_validator(mode="before")
    @classmethod
    def track_batch_size_specified(cls, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data, dict):
            data["batch_size_is_specified"] = "batch_size" in data and data["batch_size"] is not None
        return data
