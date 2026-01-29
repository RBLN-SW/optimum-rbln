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


class RBLNBlip2VisionModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLNBlip2VisionModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized BLIP-2 vision encoder models for multimodal tasks.
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")


class RBLNBlip2QFormerModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLNBlip2QFormerModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized BLIP-2 Q-Former models that bridge vision and language modalities.
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        num_query_tokens: Optional[int] = None,
        image_text_hidden_size: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            num_query_tokens (Optional[int]): The number of query tokens passed through the Transformer.
            image_text_hidden_size (Optional[int]): Dimensionality of the hidden state of the image-text fusion layer.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.num_query_tokens = num_query_tokens
        self.image_text_hidden_size = image_text_hidden_size


class RBLNBlip2ForConditionalGenerationConfig(RBLNModelConfig):
    """
    Configuration class for RBLNBlip2ForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized BLIP-2 models for conditional generation tasks that involve both image and text inputs.
    """

    submodules = ["vision_model", "qformer", "language_model"]

    def __init__(
        self,
        batch_size: Optional[int] = None,
        vision_model: Optional[RBLNModelConfig] = None,
        qformer: Optional[RBLNModelConfig] = None,
        language_model: Optional[RBLNModelConfig] = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            vision_model (Optional[RBLNModelConfig]): Configuration for the vision encoder component.
            qformer (Optional[RBLNModelConfig]): Configuration for the RBLN-optimized BLIP-2 Q-Former model.
            language_model (Optional[RBLNModelConfig]): Configuration for the language model component.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        if self.batch_size != 1:
            logger.warning("Ignore batch_size for Blip2 vision model. It will be set to 1.")
            logger.warning("Ignore batch_size for Blip2 qformer. It will be set to 1.")

        self.vision_model = self.initialize_submodule_config(
            submodule_config=vision_model, batch_size=1, force_kwargs=True
        )
        self.qformer = self.initialize_submodule_config(submodule_config=qformer, batch_size=1, force_kwargs=True)
        self.language_model = self.initialize_submodule_config(submodule_config=language_model)
