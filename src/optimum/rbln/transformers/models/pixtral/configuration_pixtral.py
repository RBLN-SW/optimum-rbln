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


class RBLNPixtralVisionModelConfig(RBLNModelConfig):
    def __init__(
        self,
        max_image_size: tuple = None,
        batch_size: int | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            max_image_size (Tuple): The size of max input images. A tuple (max_height, max_width)
            batch_size (Optional[int]): The batch size for image processing. Defaults to 1.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.max_image_size = max_image_size
        self.output_hidden_states = output_hidden_states
