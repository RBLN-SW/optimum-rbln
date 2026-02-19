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


class RBLNIdefics3VisionTransformerConfig(RBLNModelConfig):
    """
    Configuration class for RBLNIdefics3VisionTransformer.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Idefics3 vision transformer.
    """

    batch_size: PositiveIntDefaultOne = Field(default=1, description="The batch size for inference.")


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
    _submodule_hf_resolution: ClassVar[dict[str, tuple[str, str]]] = {
        "text_model": ("text_config", "language_model"),
    }

    batch_size: PositiveIntDefaultOne = Field(default=1, description="The batch size for inference.")
    vision_model: RBLNModelConfig | None = Field(
        default=None,
        description="Configuration for the vision transformer component. Includes settings specific to the vision encoder.",
    )
    text_model: dict[str, Any] | RBLNModelConfig | None = Field(
        default=None,
        description="Configuration for the text model component. Includes settings specific to the language model.",
    )

    def __init__(self, **data: Any):
        super().__init__(**data)

        if self.batch_size != 1:
            logger.warning("Ignore batch_size for Idefics3 vision transformer. It will be set to 1.")

        # initialize_submodule_config handles None, dict, and RBLNModelConfig.
        # kwargs (batch_size=1) always take priority.
        self.vision_model = self.initialize_submodule_config(
            submodule_name="vision_model", submodule_config=self.vision_model, batch_size=1
        )
        # text_model varies by model, so we use initialize_submodule_config for dict conversion
        if self.text_model is None or isinstance(self.text_model, dict):
            self.text_model = self.initialize_submodule_config(submodule_config=self.text_model)
