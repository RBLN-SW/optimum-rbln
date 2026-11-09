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
from ..models import (
    RBLNKandinsky6SRLatentUpscalerBankConfig,
    RBLNKandinsky6SRTransformer3DModelConfig,
    RBLNKandinsky6SRVAEConfig,
)


class RBLNKandinsky6SRPipelineConfig(RBLNModelConfig):
    """Config for the Kandinsky 6 video super-resolution pipeline."""

    submodules = ["transformer", "vae", "latent_upscaler"]
    subclass_non_save_attributes = ["_video_size", "_num_frames"]

    def __init__(
        self,
        transformer: RBLNKandinsky6SRTransformer3DModelConfig | None = None,
        vae: RBLNKandinsky6SRVAEConfig | None = None,
        latent_upscaler: RBLNKandinsky6SRLatentUpscalerBankConfig | None = None,
        *,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        resolution_scale: float | None = None,
        tiles_batch_size: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            transformer (RBLNKandinsky6SRTransformer3DModelConfig | None): Configuration for the transformer.
            vae (RBLNKandinsky6SRVAEConfig | None): Configuration for the VAE.
            latent_upscaler (RBLNKandinsky6SRLatentUpscalerBankConfig | None): Configuration for the latent
                upscaler bank.
            height (int | None): Height of the input video. Like diffusers, the pipeline rounds it down to a
                multiple of the VAE spatial factor. Defaults to 512.
            width (int | None): Width of the input video, rounded down like `height`. Defaults to 768.
            num_frames (int | None): Number of frames of the input video, `1 + k * 4`. Defaults to 121, the
                5-second clips the model was trained on.
            resolution_scale (float | None): Total spatial upscale: `2`, `2.25` or `4`, as `resolution_scale`
                of the diffusers pipeline call. Defaults to the diffusers default, 2.25.
            tiles_batch_size (int | None): Number of tiles one transformer call refines, as `tiles_batch_size`
                of the diffusers pipeline call. Defaults to 1.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If only one of height and width is provided.
        """
        super().__init__(**kwargs)
        if (height is None) != (width is None):
            raise ValueError("Both height and width must be provided together if used")

        self._video_size = (height or 512, width or 768)
        self._num_frames = num_frames or 121
        if resolution_scale is None:
            resolution_scale = self.get_default_values_for_original_cls("__call__", ["resolution_scale"])[
                "resolution_scale"
            ]

        self.transformer = self.initialize_submodule_config(
            transformer,
            cls_name="RBLNKandinsky6SRTransformer3DModelConfig",
            batch_size=tiles_batch_size,
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNKandinsky6SRVAEConfig",
            force_kwargs=True,
            batch_size=1,
            use_slicing=True,
            uses_encoder=True,
        )
        self.latent_upscaler = self.initialize_submodule_config(
            latent_upscaler,
            cls_name="RBLNKandinsky6SRLatentUpscalerBankConfig",
            resolution_scale=resolution_scale,
        )

    @property
    def resolution_scale(self) -> float:
        return self.latent_upscaler.resolution_scale

    @property
    def tiles_batch_size(self) -> int:
        return self.transformer.batch_size

    @property
    def video_size(self) -> tuple[int, int]:
        return self._video_size

    @property
    def num_frames(self) -> int:
        return self._num_frames


__all__ = [
    "RBLNKandinsky6SRPipelineConfig",
]
