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

from typing import Any, Optional, Tuple, Union

from ....configuration_utils import RBLNModelConfig


class RBLNQwenImageTransformer2DModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN QwenImageTransformer2DModel.

    This class provides configuration options for the transformer model used in
    the Qwen-Image-Edit diffusion pipeline.
    """

    subclass_non_save_attributes = ["_batch_size_is_specified", "_sample_size"]
    _sample_size = None

    def __init__(
        self,
        batch_size: Optional[int] = None,
        prompt_embed_length: Optional[int] = None,
        num_img_groups: int = 2,
        tensor_parallel_size: int = 1,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            sample_size (Optional[Union[int, Tuple[int, int]]]): The spatial dimensions (height, width)
                of the latent samples. If an integer is provided, it's used for both height and width.
            prompt_embed_length (Optional[int]): The length of the embedded prompt vectors.
            num_img_groups (int): The number of image shape groups passed to the transformer
                for RoPE computation. In QwenImageEdit, this is typically 2 (one for the
                target latent shape and one for the source image latent shape). Defaults to 2.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.
        """
        super().__init__(**kwargs)
        self._batch_size_is_specified = batch_size is not None

        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.prompt_embed_length = prompt_embed_length
        self.num_img_groups = num_img_groups
        self.tensor_parallel_size = tensor_parallel_size

    @property
    def batch_size_is_specified(self):
        return self._batch_size_is_specified
