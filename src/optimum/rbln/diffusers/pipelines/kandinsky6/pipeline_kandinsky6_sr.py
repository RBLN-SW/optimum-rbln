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

from dataclasses import dataclass

from diffusers import Kandinsky6SRPipeline

from ...configurations import RBLNKandinsky6SRPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


@dataclass(frozen=True)
class Kandinsky6SRTileLayout:
    """
    The shapes `Kandinsky6SRPipeline.__call__` works with for one input video size and `resolution_scale`.

    Attributes:
        video_size: The (height, width) of the video the VAE encodes, after the 1.125x pre-resize of
            `resolution_scale=2.25`.
        tile_size: The (height, width) in pixels of a tile cut from that video.
        base_tile_size: The (height, width) in pixels of a refined tile, a size the transformer was trained on.
        tiling_scale: The scale of the latent upscaler, `base_tile_size / tile_size`.
        num_latent_frames: The number of latent frames of the video.
        vae_scale_factor_spatial: The spatial compression of the VAE.
    """

    video_size: tuple[int, int]
    tile_size: tuple[int, int]
    base_tile_size: tuple[int, int]
    tiling_scale: int
    num_latent_frames: int
    vae_scale_factor_spatial: int

    @property
    def latent_tile_size(self) -> tuple[int, int]:
        height, width = self.tile_size
        return height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial

    @property
    def latent_base_tile_size(self) -> tuple[int, int]:
        height, width = self.base_tile_size
        return height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial


class RBLNKandinsky6SRPipeline(RBLNDiffusionMixin, Kandinsky6SRPipeline):
    """
    RBLN-accelerated implementation of the Kandinsky 6 video super-resolution pipeline.

    The transformer, the K-VAE and the latent upscaler are compiled for RBLN NPUs for one input video size, frame
    count and `resolution_scale`. The tiling, the PiFlow scheduler and the blending of the refined tiles run on the
    host as in diffusers.
    """

    original_class = Kandinsky6SRPipeline
    _rbln_config_class = RBLNKandinsky6SRPipelineConfig
    _submodules = ["transformer", "vae", "latent_upscaler"]

    def handle_additional_kwargs(self, **kwargs):
        # If there is no resolution_scale or tiles_batch_size of kwargs,
        # it is filled based on the compiled value.
        if kwargs.get("resolution_scale") is None:
            kwargs["resolution_scale"] = self.latent_upscaler.rbln_config.resolution_scale
        if kwargs.get("tiles_batch_size") is None:
            kwargs["tiles_batch_size"] = self.transformer.rbln_config.batch_size
        return kwargs

    def get_tile_layout(self, rbln_config: RBLNKandinsky6SRPipelineConfig) -> Kandinsky6SRTileLayout:
        """The shapes of `__call__` for the input video of `rbln_config`, computed as diffusers computes them."""
        snap = self.vae_scale_factor_spatial
        temporal = self.vae_scale_factor_temporal
        resolution_scale = rbln_config.resolution_scale
        if (rbln_config.num_frames - 1) % temporal != 0:
            raise ValueError(f"num_frames must be `1 + k * {temporal}`, got {rbln_config.num_frames}")

        height, width = (size - size % snap for size in rbln_config.video_size)
        tiling_scale, pre_upscale = (2, 1.125) if resolution_scale == 2.25 else (int(resolution_scale), 1.0)
        if pre_upscale != 1.0:
            height, width = round(height * pre_upscale / snap) * snap, round(width * pre_upscale / snap) * snap

        base_height, base_width = min(self.transformer_tile_sizes, key=lambda hw: abs(hw[1] / hw[0] - width / height))
        if base_height % (snap * tiling_scale) or base_width % (snap * tiling_scale):
            raise ValueError(
                f"The tile size {(base_height, base_width)} must be divisible by the VAE spatial factor times the"
                f" upscale factor ({snap * tiling_scale})."
            )
        tile_height, tile_width = base_height // tiling_scale, base_width // tiling_scale
        if height < tile_height or width < tile_width:
            raise ValueError(
                f"The video must be at least {tile_height}x{tile_width} pixels for `resolution_scale={resolution_scale}`,"
                f" got {height}x{width}."
            )
        return Kandinsky6SRTileLayout(
            video_size=(height, width),
            tile_size=(tile_height, tile_width),
            base_tile_size=(base_height, base_width),
            tiling_scale=tiling_scale,
            num_latent_frames=(rbln_config.num_frames - 1) // temporal + 1,
            vae_scale_factor_spatial=snap,
        )


__all__ = [
    "RBLNKandinsky6SRPipeline",
]
