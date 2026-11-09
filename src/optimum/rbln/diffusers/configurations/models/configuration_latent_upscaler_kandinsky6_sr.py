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


class RBLNKandinsky6SRLatentUpscalerBankConfig(RBLNModelConfig):
    """Configuration class for the RBLN latent upscaler bank of the Kandinsky 6 super-resolution pipeline."""

    def __init__(
        self,
        batch_size: int | None = None,
        resolution_scale: float | None = None,
        num_latent_frames: int | None = None,
        sample_size: tuple[int, int] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size of the compiled graph. Defaults to 1.
            resolution_scale (float | None): The `resolution_scale` of the pipeline call this upscaler serves:
                `2` and `2.25` compile the x2 upscaler of the bank (`2.25` resizes the video by 1.125 before
                it), `4` compiles the x4 upscaler. Filled by the pipeline config; required for a standalone
                compile.
            num_latent_frames (int | None): The number of latent frames of a tile. Filled by the pipeline
                config; required for a standalone compile.
            sample_size (tuple[int, int] | None): The latent (height, width) of a tile before upscaling.
                Filled by the pipeline config; required for a standalone compile.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer or resolution_scale is not 2, 2.25 or 4.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")
        if resolution_scale is not None and resolution_scale not in (2, 2.25, 4):
            raise ValueError(f"resolution_scale must be 2, 2.25 or 4, got {resolution_scale}")

        self.resolution_scale = resolution_scale
        self.num_latent_frames = num_latent_frames
        self.sample_size = tuple(sample_size) if sample_size is not None else None

    @property
    def scale(self) -> int | None:
        """The scale of the compiled upscaler: 4 for `resolution_scale=4`, otherwise 2."""
        if self.resolution_scale is None:
            return None
        return 4 if self.resolution_scale == 4 else 2


__all__ = [
    "RBLNKandinsky6SRLatentUpscalerBankConfig",
]
