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

from pydantic import Field, model_validator

from ....configuration_utils import PositiveIntDefaultOne, RBLNModelConfig
from ....utils.logging import get_logger
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


logger = get_logger(__name__)


class RBLNGemma3ForCausalLMConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    """Configuration for RBLNGemma3ForCausalLM."""

    image_prefill_chunk_size: int | None = Field(
        default=None,
        description="The chunk size used during the prefill phase for processing images. "
        "Used when use_image_prefill is True.",
    )

    def __init__(self, **data: Any):
        # use_attention_mask and use_position_ids are always True for Gemma3
        if "use_attention_mask" not in data or data["use_attention_mask"] is None:
            data["use_attention_mask"] = True
        if "use_position_ids" not in data or data["use_position_ids"] is None:
            data["use_position_ids"] = True
        if "prefill_chunk_size" not in data or data["prefill_chunk_size"] is None:
            data["prefill_chunk_size"] = 256

        super().__init__(**data)

    @model_validator(mode="after")
    def validate_attention_position_ids(self) -> "RBLNGemma3ForCausalLMConfig":
        if not (self.use_attention_mask and self.use_position_ids):
            raise ValueError("use_attention_mask and use_position_ids must be True for RBLNGemma3ForCausalLM")
        return self


class RBLNGemma3ForConditionalGenerationConfig(RBLNModelConfig):
    """Configuration for RBLNGemma3ForConditionalGeneration."""

    submodules: ClassVar[list[str]] = ["vision_tower", "language_model"]
    submodule_config_classes: ClassVar[dict[str, str]] = {
        "language_model": "RBLNGemma3ForCausalLMConfig",
        # vision_tower is not mapped because it varies by model
    }
    _submodule_hf_resolution: ClassVar[dict[str, tuple[str, str]]] = {
        "vision_tower": ("vision_config", "vision"),
    }

    batch_size: PositiveIntDefaultOne = Field(default=1, description="The batch size for inference.")
    vision_tower: dict[str, Any] | RBLNModelConfig | None = Field(
        default=None, description="Configuration for the vision encoder component."
    )
    language_model: RBLNModelConfig | None = Field(
        default=None, description="Configuration for the language model component."
    )

    def __init__(self, **data: Any):
        super().__init__(**data)

        if self.batch_size != 1:
            logger.warning("Ignore batch_size for Gemma3 vision tower. It will be set to 1.")

        # vision_tower varies by model, so we use initialize_submodule_config
        # kwargs (batch_size=1) always take priority
        self.vision_tower = self.initialize_submodule_config(submodule_config=self.vision_tower, batch_size=1)
        # language_model is converted by @model_validator, but we still handle None case
        if self.language_model is None:
            self.language_model = self.initialize_submodule_config(
                submodule_name="language_model", submodule_config=None
            )

    @property
    def image_prefill_chunk_size(self):
        return self.language_model.image_prefill_chunk_size

    @property
    def prefill_chunk_size(self):
        return self.language_model.prefill_chunk_size
