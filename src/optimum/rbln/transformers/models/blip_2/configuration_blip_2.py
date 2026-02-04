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


class RBLNBlip2VisionModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLNBlip2VisionModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized BLIP-2 vision encoder models for multimodal tasks.
    """

    batch_size: PositiveIntDefaultOne = Field(default=1, description="The batch size for inference.")


class RBLNBlip2QFormerModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLNBlip2QFormerModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized BLIP-2 Q-Former models that bridge vision and language modalities.
    """

    batch_size: PositiveIntDefaultOne = Field(default=1, description="The batch size for inference.")
    num_query_tokens: int | None = Field(
        default=None, description="The number of query tokens passed through the Transformer."
    )
    image_text_hidden_size: int | None = Field(
        default=None, description="Dimensionality of the hidden state of the image-text fusion layer."
    )


class RBLNBlip2ForConditionalGenerationConfig(RBLNModelConfig):
    """
    Configuration class for RBLNBlip2ForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized BLIP-2 models for conditional generation tasks that involve both image and text inputs.
    """

    submodules: ClassVar[list[str]] = ["vision_model", "qformer", "language_model"]
    submodule_config_classes: ClassVar[dict[str, str]] = {
        "vision_model": "RBLNBlip2VisionModelConfig",
        "qformer": "RBLNBlip2QFormerModelConfig",
        # language_model is not mapped because it varies by model (e.g., OPT, T5, etc.)
    }

    batch_size: PositiveIntDefaultOne = Field(default=1, description="The batch size for inference.")
    vision_model: RBLNModelConfig | None = Field(
        default=None, description="Configuration for the vision encoder component."
    )
    qformer: RBLNModelConfig | None = Field(
        default=None, description="Configuration for the RBLN-optimized BLIP-2 Q-Former model."
    )
    language_model: dict[str, Any] | RBLNModelConfig | None = Field(
        default=None, description="Configuration for the language model component."
    )

    def __init__(self, **data: Any):
        super().__init__(**data)

        if self.batch_size != 1:
            logger.warning("Ignore batch_size for Blip2 vision model. It will be set to 1.")
            logger.warning("Ignore batch_size for Blip2 qformer. It will be set to 1.")

        # initialize_submodule_config handles None, dict, and RBLNModelConfig.
        # kwargs (batch_size=1) always take priority.
        self.vision_model = self.initialize_submodule_config(
            submodule_name="vision_model", submodule_config=self.vision_model, batch_size=1
        )
        self.qformer = self.initialize_submodule_config(
            submodule_name="qformer", submodule_config=self.qformer, batch_size=1
        )
        if self.language_model is None or isinstance(self.language_model, dict):
            self.language_model = self.initialize_submodule_config(submodule_config=self.language_model)
