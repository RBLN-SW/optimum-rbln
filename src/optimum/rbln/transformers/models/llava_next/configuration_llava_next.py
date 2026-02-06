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

from ....configuration_utils import PositiveIntDefaultOne, RBLNModelConfig
from ....utils.logging import get_logger


logger = get_logger(__name__)


class RBLNLlavaNextForConditionalGenerationConfig(RBLNModelConfig):
    """
    Configuration class for RBLNLlavaNextForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized LLaVA-Next models for multimodal conditional generation tasks
    that combine vision and language processing capabilities.
    """

    submodules: ClassVar[list[str]] = ["vision_tower", "language_model"]
    # Note: vision_tower and language_model are not mapped because they vary by model

    batch_size: PositiveIntDefaultOne = Field(default=1, description="The batch size for inference.")
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
        # Handle batch_size warning and force vision_tower batch_size=1
        batch_size = data.get("batch_size", 1)
        if batch_size != 1:
            logger.warning("Ignore batch_size for LlavaNext vision tower. It will be set to 1.")

        super().__init__(**data)

        # vision_tower and language_model vary by model, so we use initialize_submodule_config
        # kwargs always take priority
        self.vision_tower = self.initialize_submodule_config(
            submodule_config=self.vision_tower,
            batch_size=1,  # vision_tower batch_size is always 1 in LlavaNext
            output_hidden_states=True,  # LlavaNext requires output_hidden_states to be True
        )

        if self.language_model is None or isinstance(self.language_model, dict):
            self.language_model = self.initialize_submodule_config(
                submodule_config=self.language_model,
            )
