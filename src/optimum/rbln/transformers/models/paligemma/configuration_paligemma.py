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


class RBLNPaliGemmaForConditionalGenerationConfig(RBLNModelConfig):
    """
    Configuration class for RBLNPaliGemmaForConditionalGenerationConfig.
    This configuration class stores the configuration parameters specific to
    RBLN-optimized PaliGemma models for multimodal conditional generation tasks
    that combine vision and language processing capabilities.
    """

    submodules: ClassVar[list[str]] = ["vision_tower", "language_model"]
    # Note: vision_tower and language_model are not mapped because they vary by model
    _allow_no_compile_cfgs = True

    batch_size: int = 1
    vision_tower: dict[str, Any] | RBLNModelConfig | None = None
    language_model: dict[str, Any] | RBLNModelConfig | None = None
    output_hidden_states: bool = False

    def __init__(self, **data: Any):
        super().__init__(**data)

        if self.batch_size != 1:
            logger.warning("Ignore batch_size for PaliGemma vision tower. It will be set to 1.")

        # vision_tower and language_model vary by model, so we use initialize_submodule_config
        if self.vision_tower is None or isinstance(self.vision_tower, dict):
            self.vision_tower = self.initialize_submodule_config(
                submodule_config=self.vision_tower,
                batch_size=1,  # vision_tower batch_size is always 1 in PaliGemma
                force_kwargs=True,
            )
        if self.language_model is None or isinstance(self.language_model, dict):
            self.language_model = self.initialize_submodule_config(
                submodule_config=self.language_model,
                batch_size=self.batch_size,
                use_position_ids=True,
                use_attention_mask=True,
                use_inputs_embeds=True,
            )

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        if v is None:
            return 1
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")
        return v


class RBLNPaliGemmaModelConfig(RBLNModelConfig):
    submodules: ClassVar[list[str]] = ["vision_tower", "language_model"]
    # Note: vision_tower and language_model are not mapped because they vary by model
    _allow_no_compile_cfgs = True

    batch_size: int = 1
    vision_tower: dict[str, Any] | RBLNModelConfig | None = None
    language_model: dict[str, Any] | RBLNModelConfig | None = None
    output_hidden_states: bool = False

    def __init__(self, **data: Any):
        super().__init__(**data)

        if self.batch_size != 1:
            logger.warning("Ignore batch_size for PaliGemma vision tower. It will be set to 1.")

        # vision_tower and language_model vary by model, so we use initialize_submodule_config
        if self.vision_tower is None or isinstance(self.vision_tower, dict):
            self.vision_tower = self.initialize_submodule_config(
                submodule_config=self.vision_tower,
                batch_size=1,  # vision_tower batch_size is always 1 in PaliGemma
                force_kwargs=True,
            )
        if self.language_model is None or isinstance(self.language_model, dict):
            self.language_model = self.initialize_submodule_config(
                submodule_config=self.language_model,
                batch_size=self.batch_size,
                use_position_ids=True,
                use_attention_mask=True,
                use_inputs_embeds=True,
                output_hidden_states=self.output_hidden_states,
            )

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        if v is None:
            return 1
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")
        return v
