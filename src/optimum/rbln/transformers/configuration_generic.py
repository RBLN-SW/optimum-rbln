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

from typing import Any, ClassVar

from pydantic import Field

from ..configuration_utils import PositiveIntDefaultOne, RBLNModelConfig


class RBLNTransformerEncoderConfig(RBLNModelConfig):
    """Base configuration for transformer encoder models."""

    rbln_model_input_names: ClassVar[list[str] | None] = None

    max_seq_len: int | None = Field(default=None, description="Maximum sequence length for the model.")
    batch_size: PositiveIntDefaultOne = Field(default=1, description="The batch size for inference.")
    model_input_names: list[str] | None = Field(default=None, description="Names of the model inputs.")
    model_input_shapes: list[tuple[int, int]] | None = Field(
        default=None, description="Shapes of the model inputs as (batch_size, seq_len)."
    )

    def __init__(self, **data: Any):
        # Set default model_input_names from class variable if not provided
        if "model_input_names" not in data or data["model_input_names"] is None:
            data["model_input_names"] = self.__class__.rbln_model_input_names
        super().__init__(**data)


class RBLNImageModelConfig(RBLNModelConfig):
    """Base configuration for image models."""

    image_size: int | tuple[int, int] | dict[str, int] | None = Field(
        default=None,
        description="The size of input images. Can be an integer for square images, "
        "a tuple (height, width), or a dict with 'height' and 'width' keys.",
    )
    batch_size: PositiveIntDefaultOne = Field(default=1, description="The batch size for inference.")

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


class RBLNModelForQuestionAnsweringConfig(RBLNTransformerEncoderConfig):
    """Configuration for question answering models."""

    pass


class RBLNModelForSequenceClassificationConfig(RBLNTransformerEncoderConfig):
    """Configuration for sequence classification models."""

    pass


class RBLNModelForMaskedLMConfig(RBLNTransformerEncoderConfig):
    """Configuration for masked language modeling."""

    pass


class RBLNModelForTextEncodingConfig(RBLNTransformerEncoderConfig):
    """Configuration for text encoding models."""

    pass


class RBLNTransformerEncoderForFeatureExtractionConfig(RBLNTransformerEncoderConfig):
    """Configuration for feature extraction models."""

    pass


class RBLNModelForImageClassificationConfig(RBLNImageModelConfig):
    """Configuration for image classification models."""

    pass


class RBLNModelForDepthEstimationConfig(RBLNImageModelConfig):
    """Configuration for depth estimation models."""

    pass
