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

from pydantic import field_validator

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger


logger = get_logger(__name__)


class RBLNIdefics3VisionTransformerConfig(RBLNModelConfig):
    """
    Configuration class for RBLNIdefics3VisionTransformer.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Idefics3 vision transformer.
    """

    batch_size: int = 1

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        if v is None:
            return 1
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")
        return v


class RBLNIdefics3ForConditionalGenerationConfig(RBLNModelConfig):
    """
    Configuration class for RBLNIdefics3ForConditionalGeneration models.

    This class extends `RBLNModelConfig` to include settings specific to the Idefics3 vision-language model optimized for RBLN devices.
    It allows configuration of the batch size and separate configurations for the vision and text submodules.
    """

    submodules: ClassVar[list[str]] = ["vision_model", "text_model"]
    submodule_config_classes: ClassVar[dict[str, str]] = {
        "vision_model": "RBLNIdefics3VisionTransformerConfig",
        # text_model is not mapped because it varies by model
    }

    batch_size: int = 1
    vision_model: RBLNModelConfig | None = None
    text_model: dict[str, Any] | RBLNModelConfig | None = None

    def __init__(self, **data: Any):
        super().__init__(**data)

        if self.batch_size != 1:
            logger.warning("Ignore batch_size for Idefics3 vision transformer. It will be set to 1.")

        # vision_model is converted by @model_validator, but we still handle None case
        if self.vision_model is None:
            self.vision_model = self.initialize_submodule_config(
                submodule_name="vision_model", submodule_config=None, batch_size=1, force_kwargs=True
            )
        # text_model varies by model, so we use initialize_submodule_config for dict conversion
        if self.text_model is None or isinstance(self.text_model, dict):
            self.text_model = self.initialize_submodule_config(submodule_config=self.text_model)

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        if v is None:
            return 1
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")
        return v
