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

from pydantic import field_validator

from ....configuration_utils import RBLNModelConfig


class RBLNCLIPTextModelConfig(RBLNModelConfig):
    """Configuration class for RBLNCLIPTextModel."""

    batch_size: int = 1

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        if v is None:
            return 1
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")
        return v


class RBLNCLIPTextModelWithProjectionConfig(RBLNCLIPTextModelConfig):
    """
    Configuration class for RBLNCLIPTextModelWithProjection.

    This configuration inherits from RBLNCLIPTextModelConfig and stores
    configuration parameters for CLIP text models with projection layers.
    """


class RBLNCLIPVisionModelConfig(RBLNModelConfig):
    """Configuration class for RBLNCLIPVisionModel."""

    batch_size: int = 1
    image_size: int | tuple[int, int] | dict[str, int] | None = None
    interpolate_pos_encoding: bool = False
    output_hidden_states: bool | None = None
    output_attentions: bool | None = None

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        if v is None:
            return 1
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")
        return v

    @property
    def image_width(self) -> int | None:
        if self.image_size is None:
            return None
        if isinstance(self.image_size, int):
            return self.image_size
        elif isinstance(self.image_size, (list, tuple)):
            return self.image_size[1]
        else:
            return self.image_size["width"]

    @property
    def image_height(self) -> int | None:
        if self.image_size is None:
            return None
        if isinstance(self.image_size, int):
            return self.image_size
        elif isinstance(self.image_size, (list, tuple)):
            return self.image_size[0]
        else:
            return self.image_size["height"]


class RBLNCLIPVisionModelWithProjectionConfig(RBLNCLIPVisionModelConfig):
    """
    Configuration class for RBLNCLIPVisionModelWithProjection.

    This configuration inherits from RBLNCLIPVisionModelConfig and stores
    configuration parameters for CLIP vision models with projection layers.
    """
