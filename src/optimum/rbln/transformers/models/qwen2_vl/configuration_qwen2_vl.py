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

from ....configuration_utils import RBLNModelConfig
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig, RBLNDecoderOnlyModelForCausalLMConfig


class RBLNQwen2VLForConditionalGenerationConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    """
    Configuration class for RBLNQwen2VLForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Qwen2-VL models for multimodal conditional generation tasks.
    """

    submodules: ClassVar[list[str]] = ["visual"]
    submodule_config_classes: ClassVar[dict[str, str]] = {"visual": "RBLNQwen2VisionTransformerPretrainedModelConfig"}

    visual: RBLNModelConfig | None = Field(default=None, description="Configuration for the vision encoder component.")

    def __init__(self, **data: Any):
        if "use_inputs_embeds" not in data or data["use_inputs_embeds"] is None:
            data["use_inputs_embeds"] = True
        super().__init__(**data)
        # visual is converted by @model_validator for dicts, but we still handle None case
        if self.visual is None:
            self.visual = self.initialize_submodule_config(submodule_name="visual", submodule_config=None)

    @model_validator(mode="after")
    def validate_use_inputs_embeds(self) -> "RBLNQwen2VLForConditionalGenerationConfig":
        if not self.use_inputs_embeds:
            raise ValueError(
                "RBLNQwen2VLForConditionalGenerationConfig does not allow `use_inputs_embeds` to be set to False, "
                "as RBLNQwen2VLForConditionalGeneration accepts only `inputs_embeds` as input."
            )
        return self


class RBLNQwen2VLModelConfig(RBLNDecoderOnlyModelConfig):
    """
    Configuration class for RBLNQwen2VLModel.
    """

    submodules: ClassVar[list[str]] = ["visual"]
    submodule_config_classes: ClassVar[dict[str, str]] = {"visual": "RBLNQwen2VisionTransformerPretrainedModelConfig"}

    visual: RBLNModelConfig | None = Field(default=None, description="Configuration for the vision encoder component.")

    def __init__(self, **data: Any):
        super().__init__(**data)
        # visual is converted by @model_validator for dicts, but we still handle None case
        if self.visual is None:
            self.visual = self.initialize_submodule_config(submodule_name="visual", submodule_config=None)


class RBLNQwen2VisionTransformerPretrainedModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLNQwen2VisionTransformerPretrainedModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Qwen2-VL vision transformer models for processing images and videos.

    Max Seq Lens:
        Since Qwen2VLForConditionalGeneration performs inference on a per-image or per-frame basis,
        `max_seq_lens` should be set based on the maximum expected resolution of the input images
        or video frames. The value must be greater than or equal to the number of patches generated
        from the input image. For example, a 224x224 image with a patch size of 14 results in
        (224 / 14) * (224 / 14) = 256 patches. Therefore, `max_seq_lens` must be at least 256.
    """

    max_seq_lens: list[int] | None = Field(
        default=None,
        description="Maximum sequence lengths for Vision Transformer attention. Can be an integer "
        "or list of integers, each indicating the number of patches in a sequence for an image or video.",
    )

    def __init__(self, **data: Any):
        # Handle max_seq_lens normalization before super().__init__
        max_seq_lens = data.get("max_seq_lens")
        if max_seq_lens is not None:
            if isinstance(max_seq_lens, int):
                data["max_seq_lens"] = [max_seq_lens]
            elif isinstance(max_seq_lens, list):
                data["max_seq_lens"] = sorted(max_seq_lens, reverse=True)
        else:
            raise ValueError("'max_seq_lens' must be specified.")

        super().__init__(**data)
