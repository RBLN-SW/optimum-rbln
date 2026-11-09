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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.latent_upscaler.latent_upscaler_kandinsky6_sr import Kandinsky6SRLatentUpscalerBank

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ...configurations import RBLNKandinsky6SRLatentUpscalerBankConfig


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


class _Kandinsky6SRLatentUpscaler(torch.nn.Module):
    """The upscaler of the bank for one scale, as `Kandinsky6SRLatentUpscalerBank.forward` picks it."""

    def __init__(self, bank: Kandinsky6SRLatentUpscalerBank, scale: int) -> None:
        super().__init__()
        self.upscaler = bank._models[list(bank.config.scales).index(scale)]

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.upscaler(latents)


class RBLNKandinsky6SRLatentUpscalerBank(RBLNModel):
    """
    RBLN implementation of Kandinsky6SRLatentUpscalerBank for Kandinsky 6 video super-resolution.

    The graph holds the upscaler of the bank for the scale of the compiled `resolution_scale`, on one latent tile
    shape.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    auto_model_class = Kandinsky6SRLatentUpscalerBank
    hf_library_name = "diffusers"
    _rbln_config_class = RBLNKandinsky6SRLatentUpscalerBankConfig
    _output_class = DecoderOutput

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNKandinsky6SRLatentUpscalerBankConfig
    ) -> torch.nn.Module:
        return _Kandinsky6SRLatentUpscaler(model, rbln_config.scale).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        layout = pipe.get_tile_layout(rbln_config)
        if rbln_config.latent_upscaler.num_latent_frames is None:
            rbln_config.latent_upscaler.num_latent_frames = layout.num_latent_frames
        if rbln_config.latent_upscaler.sample_size is None:
            rbln_config.latent_upscaler.sample_size = layout.latent_tile_size
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor] | None,
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNKandinsky6SRLatentUpscalerBankConfig,
    ) -> RBLNKandinsky6SRLatentUpscalerBankConfig:
        missing = [
            key
            for key in ("resolution_scale", "num_latent_frames", "sample_size")
            if getattr(rbln_config, key) is None
        ]
        if missing:
            raise ValueError(
                f"{', '.join(f'`{key}`' for key in missing)} must be specified to compile the Kandinsky 6 SR latent "
                "upscaler. Compiling through a pipeline fills them from the input video; a standalone compile must "
                "pass them in rbln_config."
            )
        if rbln_config.scale not in model_config.scales:
            raise ValueError(
                f"`resolution_scale={rbln_config.resolution_scale}` needs the x{rbln_config.scale} latent upscaler, "
                f"but the bank provides {list(model_config.scales)}."
            )

        height, width = rbln_config.sample_size
        input_info = [
            (
                "latents",
                [rbln_config.batch_size, model_config.in_channels, rbln_config.num_latent_frames, height, width],
                rbln_config.dtype,
            )
        ]
        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def forward(self, latents: torch.Tensor, scale: int, return_dict: bool = True) -> DecoderOutput | tuple:
        """
        Forward pass for the RBLN-optimized Kandinsky6SRLatentUpscalerBank.

        Args:
            latents (torch.Tensor): K-VAE latents scaled by `scaling_factor`, shaped
                (batch, channels, num_frames, height, width).
            scale (int): Spatial upscale factor; the scale the bank was compiled for.
            return_dict (bool): Whether to return a `DecoderOutput` instead of a plain tuple.

        Returns:
            (`~diffusers.models.autoencoders.vae.DecoderOutput` | tuple)
        """
        if scale != self.rbln_config.scale:
            raise ValueError(
                f"The latent upscaler is compiled for x{self.rbln_config.scale} "
                f"(`resolution_scale={self.rbln_config.resolution_scale}`), got scale {scale}."
            )
        compiled_shape = self.rbln_config.compile_cfgs[0].input_info[0][1]
        if list(latents.shape) != compiled_shape:
            raise ValueError(
                f"The latent upscaler is compiled for {tuple(compiled_shape)} latent tiles, got {tuple(latents.shape)}. "
                "The video size and frame count of the call must match the compiled ones."
            )
        return super().forward(latents.to(self.dtype).contiguous(), return_dict=return_dict)


__all__ = [
    "RBLNKandinsky6SRLatentUpscalerBank",
]
