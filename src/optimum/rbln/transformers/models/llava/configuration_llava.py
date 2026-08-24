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

from typing import Any

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger


logger = get_logger(__name__)


class RBLNLlavaForConditionalGenerationConfig(RBLNModelConfig):
    """
    Configuration class for RBLNLlavaForConditionalGenerationConfig.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized LLaVA models for multimodal conditional generation tasks
    that combine vision and language processing capabilities.
    """

    submodules = ["vision_tower", "language_model"]

    def __init__(
        self,
        batch_size: int | None = None,
        vision_tower: RBLNModelConfig | None = None,
        language_model: RBLNModelConfig | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size for inference. Defaults to 1.
            vision_tower (RBLNModelConfig | None): Configuration for the vision encoder component.
                This can include settings specific to the vision encoder, such as input resolution or other vision-related parameters.
                If not provided, default settings will be used.
            language_model (RBLNModelConfig | None): Configuration for the language model component.
                This can include settings specific to the language model, such as tensor parallelism or other text-related parameters.
                If not provided, default settings will be used.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If `batch_size` is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        if self.batch_size != 1:
            logger.warning("Ignore batch_size for Llava vision tower. It will be set to 1.")

        self.vision_tower = self.initialize_submodule_config(
            submodule_config=vision_tower,
            batch_size=1,  # vision_tower batch_size is always 1 in Llava
            output_hidden_states=True,  # Llava requires output_hidden_states to be True
            force_kwargs=True,
        )

        self.language_model = self.initialize_submodule_config(
            submodule_config=language_model,
            batch_size=batch_size,
            force_kwargs=True,
        )
