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


class RBLNAutoencoderKLConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Variational Autoencoder (VAE) models.

    This class inherits from RBLNModelConfig and provides specific configuration options
    for VAE models used in diffusion-based image generation.
    """

    def __init__(
        self,
        batch_size: int | None = None,
        sample_size: tuple[int, int] | None = None,
        uses_encoder: bool | None = None,
        vae_scale_factor: float | None = None,  # TODO: rename to scaling_factor
        in_channels: int | None = None,
        latent_channels: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            sample_size (Optional[Tuple[int, int]]): The spatial dimensions (height, width) of the input/output images.
                If an integer is provided, it's used for both height and width.
            uses_encoder (Optional[bool]): Whether to include the encoder part of the VAE in the model.
                When False, only the decoder is used (for latent-to-image conversion).
            vae_scale_factor (Optional[float]): The scaling factor between pixel space and latent space.
                Determines how much smaller the latent representations are compared to the original images.
            in_channels (Optional[int]): Number of input channels for the model.
            latent_channels (Optional[int]): Number of channels in the latent space.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.uses_encoder = uses_encoder
        self.vae_scale_factor = vae_scale_factor
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.sample_size = sample_size
        if isinstance(sample_size, int):
            self.sample_size = (sample_size, sample_size)

    @property
    def image_size(self):
        return self.sample_size

    @property
    def latent_sample_size(self):
        return (self.image_size[0] // self.vae_scale_factor, self.image_size[1] // self.vae_scale_factor)
