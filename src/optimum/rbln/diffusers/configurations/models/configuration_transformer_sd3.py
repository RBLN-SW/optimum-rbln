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


class RBLNSD3Transformer2DModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Stable Diffusion 3 Transformer models.

    This class inherits from RBLNModelConfig and provides specific configuration options
    for Transformer models used in diffusion models like Stable Diffusion 3.
    """

    subclass_non_save_attributes = ["_batch_size_is_specified"]

    def __init__(
        self,
        batch_size: int | None = None,
        sample_size: int | tuple[int, int] | None = None,
        prompt_embed_length: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size for inference. Defaults to 1.
            sample_size (int | tuple[int, int] | None): The spatial dimensions (height, width)
                of the generated samples. If an integer is provided, it's used for both height and width.
            prompt_embed_length (int | None): The length of the embedded prompt vectors that
                will be used to condition the transformer model.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self._batch_size_is_specified = batch_size is not None

        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.prompt_embed_length = prompt_embed_length
        self.sample_size = sample_size
        if isinstance(self.sample_size, int):
            self.sample_size = (self.sample_size, self.sample_size)

    @property
    def batch_size_is_specified(self):
        return self._batch_size_is_specified
