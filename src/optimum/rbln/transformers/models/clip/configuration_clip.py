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


class RBLNCLIPTextModelConfig(RBLNModelConfig):
    def __init__(self, batch_size: int | None = None, **kwargs: Any):
        """
        Args:
            batch_size (int | None): The batch size for text processing. Defaults to 1.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If `batch_size` is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")


class RBLNCLIPTextModelWithProjectionConfig(RBLNCLIPTextModelConfig):
    """
    Configuration class for RBLNCLIPTextModelWithProjection.

    This configuration inherits from RBLNCLIPTextModelConfig and stores
    configuration parameters for CLIP text models with projection layers.
    """


class RBLNCLIPVisionModelConfig(RBLNModelConfig):
    def __init__(
        self,
        batch_size: int | None = None,
        image_size: int | None = None,
        interpolate_pos_encoding: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size for image processing. Defaults to 1.
            image_size (int | None): The size of input images. Can be an integer for square images,
                a tuple/list (height, width), or a dictionary with 'height' and 'width' keys.
            interpolate_pos_encoding (bool | None): Whether or not to interpolate pre-trained position encodings. Defaults to `False`.
            output_hidden_states (bool | None): Whether or not to return the hidden states of all layers.
            output_attentions (bool | None): Whether or not to return the attentions tensors of all attention layers
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If `batch_size` is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.image_size = image_size
        self.interpolate_pos_encoding = interpolate_pos_encoding or False
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions

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


class RBLNCLIPVisionModelWithProjectionConfig(RBLNCLIPVisionModelConfig):
    """
    Configuration class for RBLNCLIPVisionModelWithProjection.

    This configuration inherits from RBLNCLIPVisionModelConfig and stores
    configuration parameters for CLIP vision models with projection layers.
    """
