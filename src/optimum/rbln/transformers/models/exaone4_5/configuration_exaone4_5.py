# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, Optional, Union

from ....configuration_utils import RBLNModelConfig
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig, RBLNDecoderOnlyModelForCausalLMConfig


class RBLNExaone4_5_ForConditionalGenerationConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    """
    Configuration class for RBLNExaone4_5_ForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized EXAONE-4.5 vision-language models for multimodal conditional
    generation tasks that combine vision and language processing capabilities.
    """

    submodules = ["visual"]

    def __init__(
        self,
        use_inputs_embeds: bool = True,
        visual: Optional[RBLNModelConfig] = None,
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
                "RBLNExaone4_5_ForConditionalGenerationConfig does not allow `use_inputs_embeds=False` "
                "because multimodal prefill requires embedding replacement."
            )
        self.visual = visual


class RBLNExaone4_5_ModelConfig(RBLNDecoderOnlyModelConfig):
    """
    Configuration class for RBLNExaone4_5_Model.
    """

    submodules = ["visual"]

    def __init__(self, visual: Optional[RBLNModelConfig] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.visual = self.initialize_submodule_config(submodule_config=visual)


class RBLNExaone4_5_VisionModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLNExaone4_5_VisionModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized EXAONE-4.5 vision transformer models with window-based attention
    mechanisms for processing images and videos.
    """

    def __init__(self, max_seq_len: Union[int, List[int]] = None, **kwargs: Any):
        """
        Args:
            max_seq_len (Optional[Union[int, List[int]]]): Maximum sequence lengths for Vision
                Transformer attention. Can be an integer or list of integers, each indicating
                the number of patches in a sequence for an image or video. For window-based
                attention, `max_seq_len` must be a multiple of `(window_size / patch_size)^2`.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If `max_seq_len` is None or not provided.
        """
        super().__init__(**kwargs)

        if max_seq_len is not None:
            if isinstance(max_seq_len, int):
                max_seq_len = [max_seq_len]
            elif isinstance(max_seq_len, list):
                max_seq_len.sort(reverse=True)
        else:
            raise ValueError("'max_seq_len' must be specified.")

        self.max_seq_len = max_seq_len
