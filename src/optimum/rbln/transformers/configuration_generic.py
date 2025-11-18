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

from typing import Any, List, Optional, Tuple, Union

from ..configuration_utils import RBLNModelConfig


class RBLNTransformerEncoderConfig(RBLNModelConfig):
    rbln_model_input_names: Optional[List[str]] = None

    def __init__(
        self,
        max_seq_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        model_input_names: Optional[List[str]] = None,
        model_input_shapes: Optional[List[Tuple[int, int]]] = None,
        **kwargs: Any,
    ):
        """
        Args:
            max_seq_len (Optional[int]): Maximum sequence length supported by the model.
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            model_input_names (Optional[List[str]]): Names of the input tensors for the model.
                Defaults to class-specific rbln_model_input_names if not provided.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.model_input_names = model_input_names or self.rbln_model_input_names
        self.model_input_shapes = model_input_shapes


class RBLNImageModelConfig(RBLNModelConfig):
    def __init__(
        self,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            image_size (Optional[Union[int, Tuple[int, int]]]): The size of input images.
                Can be an integer for square images or a tuple (height, width).
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.image_size = image_size
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

    @property
    def image_width(self):
        if isinstance(self.image_size, int):
            return self.image_size
        elif isinstance(self.image_size, (list, tuple)):
            return self.image_size[1]
        else:
            return self.image_size["width"]

    @property
    def image_height(self):
        if isinstance(self.image_size, int):
            return self.image_size
        elif isinstance(self.image_size, (list, tuple)):
            return self.image_size[0]
        else:
            return self.image_size["height"]


class RBLNModelForQuestionAnsweringConfig(RBLNTransformerEncoderConfig):
    pass


class RBLNModelForSequenceClassificationConfig(RBLNTransformerEncoderConfig):
    pass


class RBLNModelForMaskedLMConfig(RBLNTransformerEncoderConfig):
    pass


class RBLNModelForTextEncodingConfig(RBLNTransformerEncoderConfig):
    pass


# FIXME : Appropriate name ?
class RBLNTransformerEncoderForFeatureExtractionConfig(RBLNTransformerEncoderConfig):
    pass


class RBLNModelForImageClassificationConfig(RBLNImageModelConfig):
    pass


class RBLNModelForDepthEstimationConfig(RBLNImageModelConfig):
    pass
