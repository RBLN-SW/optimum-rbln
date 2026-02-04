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

from pydantic import Field, field_validator

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger


logger = get_logger(__name__)


class RBLNLlavaForConditionalGenerationConfig(RBLNModelConfig):
    """
    Configuration class for RBLNLlavaForConditionalGenerationConfig.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized LLaVA models for multimodal conditional generation tasks
    that combine vision and language processing capabilities.
    """

    submodules: ClassVar[list[str]] = ["vision_tower", "language_model"]
    # Note: vision_tower and language_model are not mapped because they vary by model

    batch_size: int = Field(default=1, description="The batch size for inference.")
    vision_tower: dict[str, Any] | RBLNModelConfig | None = Field(
        default=None,
        description="Configuration for the vision encoder component. "
        "Includes settings specific to the vision encoder such as input resolution.",
    )
    language_model: dict[str, Any] | RBLNModelConfig | None = Field(
        default=None,
        description="Configuration for the language model component. "
        "Includes settings specific to the language model such as tensor parallelism.",
    )

    def __init__(self, **data: Any):
        super().__init__(**data)

        if self.batch_size != 1:
            logger.warning("Ignore batch_size for Llava vision tower. It will be set to 1.")

        # vision_tower and language_model vary by model, so we use initialize_submodule_config
        # kwargs (batch_size=1) always take priority
        self.vision_tower = self.initialize_submodule_config(
            submodule_config=self.vision_tower,
            batch_size=1,  # vision_tower batch_size is always 1 in Llava
        )
        if self.language_model is None or isinstance(self.language_model, dict):
            self.language_model = self.initialize_submodule_config(
                submodule_config=self.language_model,
            )

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        if v is None:
            return 1
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")
        return v
