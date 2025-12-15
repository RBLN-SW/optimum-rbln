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

from typing import Any, Dict, List, Optional, Union

from ....configuration_utils import RBLNModelConfig
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig, RBLNDecoderOnlyModelForCausalLMConfig


class RBLNQwen2VLForConditionalGenerationConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    submodules = ["visual"]

    def __init__(
        self,
        use_inputs_embeds: bool = True,
        visual: Optional[RBLNModelConfig] = None,
        **kwargs: Dict[str, Any],
    ):
        """
        Args:
            use_inputs_embeds (bool): Whether or not to use `inputs_embeds` as input. Defaults to `True`.
            visual (Optional[RBLNModelConfig]): Configuration for the vision encoder component.
            kwargs: Additional arguments passed to the parent `RBLNDecoderOnlyModelForCausalLMConfig`.

        Raises:
            ValueError: If `use_inputs_embeds` is False.
            ValueError: If the visual configuration is provided but contains invalid settings, such as an invalid max_seq_lens (e.g., not a positive integer or insufficient for the expected resolution).
            ValueError: If visual is None and no default vision configuration can be inferred for the model architecture.
            ValueError: If any inherited parameters violate constraints defined in the parent class, such as batch_size not being a positive integer, prefill_chunk_size not being divisible by 64, or max_seq_len not meeting requirements for Flash Attention.
        """
        super().__init__(use_inputs_embeds=use_inputs_embeds, **kwargs)
        if not self.use_inputs_embeds:
            raise ValueError(
                "RBLNQwen2VLForConditionalGenerationConfig does not allow `use_inputs_embeds` to be set to False, "
                "as RBLNQwen2VLForConditionalGeneration accepts only `inputs_embeds` as input."
            )
        self.visual = visual


class RBLNQwen2VLModelConfig(RBLNDecoderOnlyModelConfig):
    """
    Configuration class for RBLNQwen2VLModel.
    """

    def __init__(self, visual: Optional[RBLNModelConfig] = None, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.visual = self.initialize_submodule_config(submodule_config=visual)


class RBLNQwen2VisionTransformerPretrainedModelConfig(RBLNModelConfig):
    def __init__(self, max_seq_lens: Union[int, List[int]] = None, **kwargs: Dict[str, Any]):
        """
        Args:
            max_seq_lens (Optional[Union[int, List[int]]]): Maximum sequence lengths for Vision
                Transformer attention. Can be an integer or list of integers, each indicating
                the number of patches in a sequence for an image or video. For example, an image
                of 224x224 pixels with patch size 14 results in (224/14) * (224/14) = 256 patches,
                so `max_seq_lens` must be at least 256. RBLN optimization runs inference per image
                or video frame, so set `max_seq_lens` to match the maximum expected resolution to
                optimize computation. If not provided, a `ValueError` is raised.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

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
        super().__init__(**kwargs)

        if max_seq_lens is not None:
            if isinstance(max_seq_lens, int):
                max_seq_lens = [max_seq_lens]
            elif isinstance(max_seq_lens, list):
                max_seq_lens.sort(reverse=True)
        else:
            raise ValueError("'max_seq_lens' must be specified.")

        self.max_seq_lens = max_seq_lens
