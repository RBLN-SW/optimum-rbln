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


class RBLNAutoencoderKLQwenImage21Config(RBLNModelConfig):
    """Configuration class for RBLN Qwen-Image 2.1 Variational Autoencoder (VAE) models."""

    def __init__(
        self,
        batch_size: int | None = None,
        sample_size: tuple[int, int] | list[tuple[int, int]] | None = None,
        uses_encoder: bool | None = None,
        encoder_sample_size: tuple[int, int] | list[tuple[int, int]] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size for inference. Defaults to 1.
            sample_size (tuple[int, int] | list[tuple[int, int]] | None): The (height, width) of the decoded image,
                in pixels, or a list of them; the decoder compiles one graph for each. Filled by the pipeline config;
                required for a standalone compile.
            uses_encoder (bool | None): Whether to compile the encoder, which image editing needs to encode the
                condition images. Defaults to False.
            encoder_sample_size (tuple[int, int] | list[tuple[int, int]] | None): The (height, width) of the
                encoder canvases, in pixels, or a list of them; the encoder compiles one graph for each. An image runs
                at the top left of the smallest canvas that holds it, so one canvas serves every image size up to its
                own. Defaults to `sample_size`.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        if sample_size is not None and isinstance(sample_size[0], int):
            sample_size = [sample_size]
        self.sample_size = None if sample_size is None else [(size[0], size[1]) for size in sample_size]
        self.uses_encoder = uses_encoder
        if encoder_sample_size is not None and isinstance(encoder_sample_size[0], int):
            encoder_sample_size = [encoder_sample_size]
        self.encoder_sample_size = (
            None if encoder_sample_size is None else [(size[0], size[1]) for size in encoder_sample_size]
        )

    @property
    def image_size(self):
        # The pipeline fills an omitted height and width with it, so it is set only when one size is compiled.
        return self.sample_size[0] if self.sample_size is not None and len(self.sample_size) == 1 else None


__all__ = [
    "RBLNAutoencoderKLQwenImage21Config",
]
