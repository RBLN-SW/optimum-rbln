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
from ....utils.logging import get_logger


logger = get_logger(__name__)


class RBLNAutoencoderKLWanConfig(RBLNModelConfig):
    """Configuration class for RBLN Wan Variational Autoencoder (VAE) models."""

    def __init__(
        self,
        batch_size: int | None = None,
        uses_encoder: bool | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        num_channels_latents: int | None = None,
        vae_scale_factor_temporal: int | None = None,
        vae_scale_factor_spatial: int | None = None,
        use_slicing: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            uses_encoder (Optional[bool]): Whether to include the encoder part of the VAE in the model.
                When False, only the decoder is used (for latent-to-video conversion).
            num_frames (Optional[int]): The number of frames in the generated video. Filled by the
                pipeline config (per-pipeline diffusers default); required for a standalone compile.
            height (Optional[int]): The height in pixels of the generated video. Filled by the
                pipeline config (per-pipeline diffusers default); required for a standalone compile.
            width (Optional[int]): The width in pixels of the generated video. Filled by the
                pipeline config (per-pipeline diffusers default); required for a standalone compile.
            num_channels_latents (Optional[int]): The number of channels in latent space. The pipeline
                hook sets it to the transformer's out_channels; a standalone compile falls back to the
                VAE z_dim.
            vae_scale_factor_temporal (Optional[int]): The scaling factor between time space and latent space.
                Determines how much shorter the latent representations are compared to the original videos.
            vae_scale_factor_spatial (Optional[int]): The scaling factor between pixel space and latent space.
                Determines how much smaller the latent representations are compared to the original videos.
            use_slicing (bool): Run batched requests through the batch-1 compiled graphs one sample
                at a time. Defaults to True. Batched Wan VAE graphs are not supported (the decoder's
                working set is already near the device limit), so batch_size is always normalized to 1
                and a batch_size > 1 request with use_slicing=False falls back to slicing, with a
                warning. Slicing keys off the runtime input shape, so batched calls still work.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.use_slicing = use_slicing
        # The Wan VAE decoder's working set is already near the device limit at full
        # resolution, so the graphs are always compiled at batch_size=1 and batched
        # requests run per sample (use_slicing, keyed off the runtime input shape).
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")
        elif self.batch_size > 1:
            if not self.use_slicing:
                logger.warning(
                    "Batched Wan VAE graphs are not supported for memory efficiency; "
                    "falling back to per-sample slicing."
                )
                self.use_slicing = True
            self.batch_size = 1

        self.uses_encoder = uses_encoder
        # No size defaults here: each pipeline config supplies its own diffusers-default
        # height/width/num_frames, and a standalone compile must pass them explicitly.
        self.num_frames = num_frames
        self.height = height
        self.width = width

        self.num_channels_latents = num_channels_latents
        self.vae_scale_factor_temporal = vae_scale_factor_temporal
        self.vae_scale_factor_spatial = vae_scale_factor_spatial

    @property
    def image_size(self):
        return (self.height, self.width)
