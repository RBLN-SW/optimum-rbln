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

from typing import Any, List, Union

from ....configuration_utils import RBLNModelConfig
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig, RBLNDecoderOnlyModelForCausalLMConfig


class RBLNQwen3VLForConditionalGenerationConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    """
    Configuration class for RBLNQwen3VLForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Qwen3-VL models for multimodal conditional generation tasks
    that combine vision and language processing capabilities.
    """

    # submodules = ["visual"]

    def __init__(
        self,
        use_inputs_embeds: bool = True,
        # visual: Optional[RBLNModelConfig] = None,
        **kwargs: Any,
    ):
        """
        Args:
            use_inputs_embeds (bool): Whether or not to use `inputs_embeds` as input. Defaults to `True`.
            visual (Optional[RBLNModelConfig]): Configuration for the vision encoder component.
            kwargs: Additional arguments passed to the parent `RBLNDecoderOnlyModelForCausalLMConfig`.

        Raises:
            ValueError: If `use_inputs_embeds` is False.
        """
        super().__init__(use_inputs_embeds=use_inputs_embeds, **kwargs)
        if not self.use_inputs_embeds:
            raise ValueError(
                "RBLNQwen3VLForConditionalGenerationConfig does not allow `use_inputs_embeds` to be set to False, "
                "as RBLNQwen3VLForConditionalGeneration accepts only `inputs_embeds` as input."
            )
        # self.visual = visual


class RBLNQwen3VLModelConfig(RBLNDecoderOnlyModelConfig):
    # submodules = ["visual"]

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # self.visual = self.initialize_submodule_config(submodule_config=visual)


class RBLNQwen3VLVisionModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLNQwen3VLVisionModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Qwen3-VL vision transformer models for processing images and videos.
    """

    def __init__(self, max_seq_lens: Union[int, List[int]] = None, **kwargs: Any):
        """
        Args:
            max_seq_lens (Optional[Union[int, List[int]]]): Maximum sequence lengths for Vision
                Transformer attention. Can be an integer or list of integers, each indicating
                the number of patches in a sequence for an image or video. For example, an image
                of 224x224 pixels with patch size 16 and spatial_merge_size 2 yields
                (224/16/2) * (224/16/2) = 49 merged patches. RBLN optimization runs inference
                per image or video frame, so set `max_seq_len` to match the maximum expected
                resolution to reduce computation. If not provided, a `ValueError` is raised.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If `max_seq_lens` is None or not provided.
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
