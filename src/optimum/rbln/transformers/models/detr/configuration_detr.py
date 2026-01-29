# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional, Tuple, Union

from ....configuration_utils import RBLNModelConfig


class RBLNDetrForObjectDetectionConfig(RBLNModelConfig):
    """
    Configuration class for RBLNDetrForObjectDetection.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized DETR models for object detection tasks.
    """

    def __init__(
        self,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            image_size (Optional[Union[int, Tuple[int, int]]]): The size of input images
                for compile shape. Can be an integer for square images or a tuple (height, width).
                If not provided, defaults to (1333, 1333) based on DETR's default longest_edge setting.
                Images smaller than this will be padded during inference.
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
