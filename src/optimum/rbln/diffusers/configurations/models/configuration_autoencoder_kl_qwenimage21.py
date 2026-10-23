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


def _image_sizes(size: tuple[int, int] | list[tuple[int, int]] | None) -> list[tuple[int, int]] | None:
    # One (height, width) or a list of them; a saved config holds the list form.
    if size is None:
        return None
    sizes = [size] if isinstance(size[0], int) else size
    return [(s[0], s[1]) for s in sizes]


class RBLNAutoencoderKLQwenImage21Config(RBLNModelConfig):
    """
    Configuration class for the RBLN Qwen-Image 2.1 VAE.

    The VAE works on single images. The decoder compiles one graph for every output size. The encoder, used by image
    editing, compiles one graph for every canvas size and encodes any image that fits a canvas.
    """

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
                in pixels, or a list of them. Defaults to (1024, 1024), the pipeline's default output.
            uses_encoder (bool | None): Whether to compile the encoder, which image editing needs to encode the
                condition images. Defaults to False.
            encoder_sample_size (tuple[int, int] | list[tuple[int, int]] | None): The (height, width) of the
                encoder canvases, in pixels, or a list of them. An image runs at the top left of the smallest canvas
                that holds it, so one canvas serves every image size up to its own. Defaults to `sample_size`.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.sample_size = _image_sizes(sample_size)
        self.uses_encoder = uses_encoder
        self.encoder_sample_size = _image_sizes(encoder_sample_size)


__all__ = [
    "RBLNAutoencoderKLQwenImage21Config",
]
