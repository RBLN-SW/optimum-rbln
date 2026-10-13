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


class RBLNAutoencoderKLCogVideoXConfig(RBLNModelConfig):
    """Configuration class for RBLN CogVideoX Variational Autoencoder (VAE) models."""

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
        num_latent_frames_batch_size: int | None = None,
        cache_dtype: str | None = None,
        use_slicing: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            uses_encoder (Optional[bool]): Whether to include the encoder part of the VAE in the model.
                Only the decoder is supported now, so this must be False or None.
            num_frames (Optional[int]): The number of frames in the generated video. Filled by the
                pipeline config; required for a standalone compile.
            height (Optional[int]): The height in pixels of the generated video. Filled by the
                pipeline config; required for a standalone compile.
            width (Optional[int]): The width in pixels of the generated video. Filled by the
                pipeline config; required for a standalone compile.
            num_channels_latents (Optional[int]): The number of channels in latent space. Falls back
                to the VAE latent_channels.
            vae_scale_factor_temporal (Optional[int]): The scaling factor between time space and latent space.
            vae_scale_factor_spatial (Optional[int]): The scaling factor between pixel space and latent space.
            num_latent_frames_batch_size (Optional[int]): The number of latent frames that one decoder
                call after the first one decodes. Defaults to 2, the value diffusers uses. The first call
                also takes the odd remainder frame.
            cache_dtype (Optional[str]): The dtype of the temporal conv caches that stay on the device
                between decoder calls. Defaults to "bfloat16".
            use_slicing (bool): Run batched requests one sample at a time on the batch-1 compiled graphs.
                Defaults to True; a batch_size > 1 request is normalized to 1 with a warning.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        if self.num_devices is not None and self.num_devices > 1:
            raise ValueError(
                f"RBLNAutoencoderKLCogVideoX runs on one device; got num_devices={self.num_devices}. "
                "Set num_devices on the transformer config only."
            )
        self.use_slicing = use_slicing
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")
        elif self.batch_size > 1:
            if not self.use_slicing:
                self.use_slicing = True
            logger.warning(
                "Batched CogVideoX VAE graphs are not supported for memory efficiency; "
                "falling back to per-sample slicing."
            )
            self.batch_size = 1

        if uses_encoder:
            raise NotImplementedError("RBLNAutoencoderKLCogVideoX supports the decoder only (uses_encoder=False).")
        self.uses_encoder = uses_encoder
        self.num_frames = num_frames
        self.height = height
        self.width = width

        self.num_channels_latents = num_channels_latents
        self.vae_scale_factor_temporal = vae_scale_factor_temporal
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        self.num_latent_frames_batch_size = num_latent_frames_batch_size or 2
        self.cache_dtype = cache_dtype or "bfloat16"

    @property
    def image_size(self):
        return (self.height, self.width)

    @property
    def num_latent_frames(self) -> int:
        if self.num_frames is None or self.vae_scale_factor_temporal is None:
            raise ValueError("`num_frames` and `vae_scale_factor_temporal` must be set to get `num_latent_frames`.")
        return (self.num_frames - 1) // self.vae_scale_factor_temporal + 1

    @property
    def first_chunk_size(self) -> int:
        # Mirrors AutoencoderKLCogVideoX._decode: the first call takes the odd remainder frame too.
        batch = self.num_latent_frames_batch_size
        return min(self.num_latent_frames, batch + self.num_latent_frames % batch)


__all__ = [
    "RBLNAutoencoderKLCogVideoXConfig",
]
