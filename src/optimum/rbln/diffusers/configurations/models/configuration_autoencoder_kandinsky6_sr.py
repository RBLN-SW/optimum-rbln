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


class RBLNKandinsky6SRVAEConfig(RBLNModelConfig):
    """Configuration class for the RBLN causal video K-VAE of the Kandinsky 6 super-resolution pipeline."""

    def __init__(
        self,
        batch_size: int | None = None,
        uses_encoder: bool | None = None,
        sample_size: tuple[int, int] | None = None,
        latent_sample_size: tuple[int, int] | None = None,
        segment_frames: int | None = None,
        use_slicing: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size of the compiled graphs. Defaults to 1; the pipelines
                always compile the VAE with batch 1 and slicing on.
            uses_encoder (bool | None): Whether to include the encoder part of the VAE in the model.
                Defaults to True: the super-resolution pipeline encodes the input video.
            sample_size (tuple[int, int] | None): The (height, width) in pixels of the video the encoder
                takes. Filled by the pipeline config; required for a standalone compile with the encoder.
            latent_sample_size (tuple[int, int] | None): The (height, width) of the latents the decoder
                takes. Filled by the pipeline config; required for a standalone compile.
            segment_frames (int | None): Pixel frames of one segment after the leading frame, a multiple of
                the temporal compression. diffusers encodes and decodes in segments of 16 frames; the causal
                convs carry their padding between segments, so a shorter segment gives the same video with
                less device memory per graph. Defaults to one temporal compression group (4 frames).
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

        self.uses_encoder = True if uses_encoder is None else uses_encoder
        self.sample_size = tuple(sample_size) if sample_size is not None else None
        self.latent_sample_size = tuple(latent_sample_size) if latent_sample_size is not None else None
        self.segment_frames = segment_frames

    @property
    def image_size(self):
        return self.sample_size


__all__ = [
    "RBLNKandinsky6SRVAEConfig",
]
