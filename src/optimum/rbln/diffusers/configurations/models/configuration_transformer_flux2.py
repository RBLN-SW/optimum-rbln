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

from typing import Any

from ....configuration_utils import RBLNModelConfig


class RBLNFlux2Transformer2DModelConfig(RBLNModelConfig):
    """Static image, prompt and reference geometry for the FLUX.2 klein transformer."""

    def __init__(
        self,
        batch_size: int = 1,
        image_size: tuple[int, int] = (1024, 1024),
        prompt_embed_length: int = 512,
        reference_image_count: int = 0,
        reference_image_size: tuple[int, int] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int): The compiled request batch size.
            image_size (tuple[int, int]): Generated image height and width in pixels.
            prompt_embed_length (int): The compiled prompt length, at most 512 tokens.
            reference_image_count (int): The exact number of reference images per request.
            reference_image_size (tuple[int, int] | None): Reference image height and width after native preprocessing. Defaults to image_size.
            kwargs: Additional arguments passed to RBLNModelConfig.
        """
        super().__init__(**kwargs)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(reference_image_count) is not int or reference_image_count < 0:
            raise ValueError("reference_image_count must be a nonnegative integer")
        if type(prompt_embed_length) is not int or not 1 <= prompt_embed_length <= 512:
            raise ValueError("prompt_embed_length must be between 1 and 512")
        self.batch_size = batch_size
        self.image_size = self._image_size(image_size)
        self.reference_image_size = self._image_size(
            image_size if reference_image_size is None else reference_image_size
        )
        self.prompt_embed_length = prompt_embed_length
        self.reference_image_count = reference_image_count

    @staticmethod
    def _image_size(value):
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 2
            or any(type(dimension) is not int or dimension < 16 or dimension % 16 for dimension in value)
        ):
            raise ValueError("Image height and width must be positive multiples of 16")
        return tuple(value)

    @property
    def image_token_count(self):
        height, width = self.image_size
        reference_height, reference_width = self.reference_image_size
        return (height // 16) * (width // 16) + self.reference_image_count * (
            (reference_height // 16) * (reference_width // 16)
        )


__all__ = ["RBLNFlux2Transformer2DModelConfig"]
