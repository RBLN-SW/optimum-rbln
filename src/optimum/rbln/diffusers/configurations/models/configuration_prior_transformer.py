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

from pydantic import PrivateAttr, field_validator

from ....configuration_utils import RBLNModelConfig


class RBLNPriorTransformerConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Prior Transformer models.

    This class inherits from RBLNModelConfig and provides specific configuration options
    for Transformer models used in diffusion models like Kandinsky V2.2.
    """

    batch_size: int = 1
    embedding_dim: int | None = None
    num_embeddings: int | None = None

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
