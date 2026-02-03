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

from pydantic import model_validator

from ....configuration_utils import RBLNModelConfig
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig, RBLNDecoderOnlyModelForCausalLMConfig


class RBLNQwen2VLForConditionalGenerationConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    submodules: ClassVar[list[str]] = ["visual"]
    submodule_config_classes: ClassVar[dict[str, str]] = {"visual": "RBLNQwen2VisionTransformerPretrainedModelConfig"}

    visual: RBLNModelConfig | None = None

    def __init__(self, **data: Any):
        """
        Args:
            use_inputs_embeds (bool): Whether or not to use `inputs_embeds` as input. Defaults to `True`.
            visual (Optional[RBLNModelConfig]): Configuration for the vision encoder component.
            **data: Additional arguments passed to the parent `RBLNDecoderOnlyModelForCausalLMConfig`.

        Raises:
            ValueError: If `use_inputs_embeds` is False.
            ValueError: If the visual configuration is provided but contains invalid settings, such as an invalid max_seq_lens (e.g., not a positive integer or insufficient for the expected resolution).
            ValueError: If visual is None and no default vision configuration can be inferred for the model architecture.
            ValueError: If any inherited parameters violate constraints defined in the parent class, such as batch_size not being a positive integer, prefill_chunk_size not being divisible by 64, or max_seq_len not meeting requirements for Flash Attention.
        """
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

    visual: RBLNModelConfig | None = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        # visual is converted by @model_validator for dicts, but we still handle None case
        if self.visual is None:
            self.visual = self.initialize_submodule_config(submodule_name="visual", submodule_config=None)


class RBLNQwen2VisionTransformerPretrainedModelConfig(RBLNModelConfig):
    max_seq_lens: list[int] | None = None

    def __init__(self, **data: Any):
        """
        Args:
            max_seq_lens (Optional[Union[int, List[int]]]): Maximum sequence lengths for Vision
                Transformer attention. Can be an integer or list of integers, each indicating
                the number of patches in a sequence for an image or video. For example, an image
                of 224x224 pixels with patch size 14 results in (224/14) * (224/14) = 256 patches,
                so `max_seq_lens` must be at least 256. RBLN optimization runs inference per image
                or video frame, so set `max_seq_lens` to match the maximum expected resolution to
                optimize computation. If not provided, a `ValueError` is raised.
            **data: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
            ValueError: If `max_seq_lens` (or any value in the list) is not a positive integer.
            ValueError: If `max_seq_lens` is insufficient for the expected image/video resolution.
            ValueError: If `batch_size` (inherited from RBLNModelConfig) is not a positive integer.

        Max Seq Lens:
            Since `Qwen2VLForConditionalGeneration` performs inference on a per-image or per-frame basis,
            `max_seq_lens` should be set based on the maximum expected resolution of the input images or video frames.

            The value must be greater than or equal to the number of patches generated from the input image.
            For example, a 224x224 image with a patch size of 14 results in (224 / 14) * (224 / 14) = 256 patches.
            Therefore, `max_seq_lens` must be at least 256.
        """
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
