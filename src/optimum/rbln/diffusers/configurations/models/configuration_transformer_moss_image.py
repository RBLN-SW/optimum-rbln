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


class RBLNMOSSImageTransformerConfig(RBLNModelConfig):
    """
    Configuration class for the RBLN MOSS-Image transformer.

    The transformer compiles for one latent size and one prompt length.
    """

    def __init__(
        self,
        batch_size: int | None = None,
        sample_size: tuple[int, int] | None = None,
        max_sequence_length: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The number of images one forward pass denoises. The pipeline runs the
                conditional and the unconditional prompts in separate passes, so this is the number of prompts times
                `num_images_per_prompt`. Defaults to 1.
            sample_size (tuple[int, int] | None): The latent (height, width): the image size divided by 16. Defaults
                to (64, 64), the pipeline's default 1024x1024 image.
            max_sequence_length (int | None): The number of text tokens the graph reads. A shorter prompt is padded
                and masked. Defaults to 512.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.sample_size = tuple(sample_size) if sample_size is not None else (64, 64)
        self.max_sequence_length = max_sequence_length or 512


__all__ = [
    "RBLNMOSSImageTransformerConfig",
]
