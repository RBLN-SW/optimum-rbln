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


class RBLNPriorTransformerConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Prior Transformer models.

    This class inherits from RBLNModelConfig and provides specific configuration options
    for Transformer models used in diffusion models like Kandinsky V2.2.
    """

    batch_size: PositiveIntDefaultOne = Field(default=1, description="The batch size for inference.")
    embedding_dim: int | None = Field(default=None, description="Dimension of the embeddings.")
    num_embeddings: int | None = Field(default=None, description="Number of embeddings.")
    batch_size_is_specified: bool = Field(
        default=False, exclude=True, description="Whether the batch size was explicitly specified by the user."
    )

    @model_validator(mode="before")
    @classmethod
    def track_batch_size_specified(cls, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data, dict):
            data["batch_size_is_specified"] = "batch_size" in data and data["batch_size"] is not None
        return data
