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


class RBLNAutoencoderKLCogVideoXConfig(RBLNModelConfig):
    """Configuration class for RBLN CogVideoX Variational Autoencoder (VAE) models."""

    def __init__(
        self,
        batch_size: int | None = None,
        uses_encoder: bool | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        num_latent_frames_batch_size: int | None = None,
        use_slicing: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size of the compiled graphs. Defaults to 1; the pipelines
                always compile the VAE with batch 1 and slicing on.
            uses_encoder (Optional[bool]): Whether to include the encoder part of the VAE in the model.
                Image-to-video uses it to encode the input image as a 1-frame video; text-to-video does not.
            num_frames (Optional[int]): The number of frames in the generated video. Filled by the
                pipeline config; required for a standalone compile.
            height (Optional[int]): The height in pixels of the generated video. Filled by the
                pipeline config; required for a standalone compile.
            width (Optional[int]): The width in pixels of the generated video. Filled by the
                pipeline config; required for a standalone compile.
            num_latent_frames_batch_size (Optional[int]): The number of latent frames per decoder call, as
                `AutoencoderKLCogVideoX.num_latent_frames_batch_size` (2 in diffusers). The first call also takes
                the remainder frames. Defaults to the value of the loaded model.
            use_slicing (bool): Run a batched request one sample at a time, as diffusers `use_slicing`
                (it splits the batch axis only). Defaults to True.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.use_slicing = use_slicing
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.uses_encoder = uses_encoder
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_latent_frames_batch_size = num_latent_frames_batch_size

    @property
    def image_size(self):
        return (self.height, self.width)


__all__ = [
    "RBLNAutoencoderKLCogVideoXConfig",
]
