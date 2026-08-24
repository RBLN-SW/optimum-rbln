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


class RBLNIdefics3VisionTransformerConfig(RBLNModelConfig):
    """
    Configuration class for RBLNIdefics3VisionTransformer.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Idefics3 vision transformer.
    """

    def __init__(
        self,
        batch_size: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")


class RBLNIdefics3ForConditionalGenerationConfig(RBLNModelConfig):
    """
    Configuration class for RBLNIdefics3ForConditionalGeneration models.

    This class extends `RBLNModelConfig` to include settings specific to the Idefics3 vision-language model optimized for RBLN devices.
    It allows configuration of the batch size and separate configurations for the vision and text submodules.

    Attributes:
        submodules (list[str]): List of submodules included in the model. Defaults to `["vision_model", "text_model"]`.
    """

    submodules = ["vision_model", "text_model"]

    def __init__(
        self,
        batch_size: int | None = None,
        vision_model: RBLNModelConfig | None = None,
        text_model: RBLNModelConfig | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size for inference. Defaults to 1.
            vision_model (RBLNModelConfig | None): Configuration for the vision transformer component.
                This can include settings specific to the vision encoder, such as input resolution or other vision-related parameters.
                If not provided, default settings will be used.
            text_model (RBLNModelConfig | None): Configuration for the text model component.
                This can include settings specific to the language model, such as tensor parallelism or other text-related parameters.
                If not provided, default settings will be used.
            kwargs: Additional arguments passed to the parent `RBLNModelConfig`.

        Raises:
            ValueError: If `batch_size` is not a positive integer.
        """

        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        if self.batch_size != 1:
            logger.warning("Ignore batch_size for Idefics3 vision transformer. It will be set to 1.")

        self.vision_model = self.initialize_submodule_config(
            submodule_config=vision_model, batch_size=1, force_kwargs=True
        )
        self.text_model = self.initialize_submodule_config(
            submodule_config=text_model, batch_size=batch_size, force_kwargs=True
        )
