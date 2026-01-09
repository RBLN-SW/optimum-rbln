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

from typing import Any, Dict, Optional

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger


logger = get_logger(__name__)


class RBLNAutoencoderKLWanConfig(RBLNModelConfig):
    """Configuration class for RBLN Wan Variational Autoencoder (VAE) models."""

    def __init__(
        self,
        batch_size: Optional[int] = None,
        uses_encoder: Optional[bool] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_channels_latents: Optional[int] = None,
        vae_scale_factor_temporal: Optional[int] = None,
        vae_scale_factor_spatial: Optional[int] = None,
        use_slicing: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            uses_encoder (Optional[bool]): Whether to include the encoder part of the VAE in the model.
                When False, only the decoder is used (for latent-to-video conversion).
            num_frames (Optional[int]): The number of frames in the generated video. Defaults to 93.
            height (Optional[int]): The height in pixels of the generated video. Defaults to 704.
            width (Optional[int]): The width in pixels of the generated video. Defaults to 1280.
            num_channels_latents (Optional[int]): The number of channels in latent space. Defaults to 24 (z_dim).
            vae_scale_factor_temporal (Optional[int]): The scaling factor between time space and latent space.
                Determines how much shorter the latent representations are compared to the original videos.
            vae_scale_factor_spatial (Optional[int]): The scaling factor between pixel space and latent space.
                Determines how much smaller the latent representations are compared to the original videos.
            use_slicing (Optional[bool]): Enable sliced VAE encoding and decoding.
                If True, the VAE will split the input tensor in slices to compute encoding or decoding in several steps.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        # Since the Wan VAE Decoder already requires significant memory,
        # Optimum-rbln cannot execute this model on RBLN NPU when the batch size > 1.
        # However, the Wan VAE Decoder supports batch slicing when the batch size is greater than 1,
        # Optimum-rbln utilizes this method by compiling with batch_size=1 to enable batch slicing.
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")
        elif self.batch_size > 1:
            logger.warning("The batch size of Wan VAE Decoder will be explicitly 1 for memory efficiency.")
            self.batch_size = 1

        self.uses_encoder = uses_encoder
        self.num_frames = num_frames or 93
        self.height = height or 704
        self.width = width or 1280

        self.num_channels_latents = num_channels_latents or 24  # z_dim default
        self.vae_scale_factor_temporal = vae_scale_factor_temporal
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        self.use_slicing = use_slicing or False

    @property
    def image_size(self):
        return (self.height, self.width)
