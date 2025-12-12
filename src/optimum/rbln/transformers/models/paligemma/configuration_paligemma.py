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

from typing import Any, Optional

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

    submodules = ["vision_tower", "language_model"]
    _allow_no_compile_cfgs = True

    def __init__(
        self,
        batch_size: Optional[int] = None,
        vision_tower: Optional[RBLNModelConfig] = None,
        language_model: Optional[RBLNModelConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            vision_tower (Optional[RBLNModelConfig]): Configuration for the vision encoder component.
                This can include settings specific to the vision encoder, such as input resolution or other vision-related parameters.
                If not provided, default settings will be used.
            language_model (Optional[RBLNModelConfig]): Configuration for the language model component.
                This can include settings specific to the language model, such as tensor parallelism or other text-related parameters.
                If not provided, default settings will be used.
            output_hidden_states (Optional[bool]): Whether to output the hidden states of the decoder. Defaults to False.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.
        Raises:
            ValueError: If `batch_size` is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        if self.batch_size != 1:
            logger.warning("Ignore batch_size for PaliGemma vision tower. It will be set to 1.")

        self.output_hidden_states = output_hidden_states or False

        self.vision_tower = self.initialize_submodule_config(
            submodule_config=vision_tower,
            batch_size=1,  # vision_tower batch_size is always 1 in PaliGemma
            force_kwargs=True,
        )
        self.language_model = self.initialize_submodule_config(
            submodule_config=language_model,
            batch_size=batch_size,
            use_position_ids=True,
            use_attention_mask=True,
            use_inputs_embeds=True,
        )


class RBLNPaliGemmaModelConfig(RBLNModelConfig):
    submodules = ["vision_tower", "language_model"]
    _allow_no_compile_cfgs = True

    def __init__(
        self,
        batch_size: Optional[int] = None,
        vision_tower: Optional[RBLNModelConfig] = None,
        language_model: Optional[RBLNModelConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            vision_tower (Optional[RBLNModelConfig]): Configuration for the vision encoder component.
                This can include settings specific to the vision encoder, such as input resolution or other vision-related parameters.
                If not provided, default settings will be used.
            language_model (Optional[RBLNModelConfig]): Configuration for the language model component.
                This can include settings specific to the language model, such as tensor parallelism or other text-related parameters.
                If not provided, default settings will be used.
            output_hidden_states (Optional[bool]): Whether to output the hidden states of the decoder. Defaults to False.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.
        Raises:
            ValueError: If `batch_size` is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        if self.batch_size != 1:
            logger.warning("Ignore batch_size for PaliGemma vision tower. It will be set to 1.")

        self.output_hidden_states = output_hidden_states or False
        self.vision_tower = self.initialize_submodule_config(
            submodule_config=vision_tower,
            batch_size=1,  # vision_tower batch_size is always 1 in PaliGemma
            force_kwargs=True,
        )

        self.language_model = self.initialize_submodule_config(
            submodule_config=language_model,
            batch_size=batch_size,
            use_position_ids=True,
            use_attention_mask=True,
            use_inputs_embeds=True,
            output_hidden_states=output_hidden_states,
        )
