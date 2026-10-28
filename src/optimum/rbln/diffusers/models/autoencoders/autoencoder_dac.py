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

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ...configurations import RBLNDACConfig
from ...pipelines.moss_soundeffect.pipeline_moss_soundeffect_upstream import DAC
from .vae import RBLNRuntimeVAEDecoder, _VAEDACDecoder


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


class RBLNConvTranspose1d(torch.nn.Module):
    def __init__(self, conv: torch.nn.ConvTranspose1d):
        super().__init__()
        if conv.groups != 1:
            raise NotImplementedError("Grouped transposed convolutions are not supported.")
        (self.stride,), (padding,), (self.output_padding,), (self.dilation,) = (
            conv.stride,
            conv.padding,
            conv.output_padding,
            conv.dilation,
        )
        if self.output_padding > padding:
            raise NotImplementedError(
                "Transposed convolutions with more output padding than padding are not supported."
            )
        # The device kernel rejects `output_padding`: pad that much less and drop as many leading samples (exact).
        self.padding = padding - self.output_padding
        in_channels, out_channels, kernel_size = conv.weight.shape
        # Kernels above about 6.4M elements fail to compile ("OOM reported as TileError"); split by output channels.
        chunk = max(1, 4_000_000 // (in_channels * kernel_size))
        weight, bias = conv.weight.detach(), None if conv.bias is None else conv.bias.detach()
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(weight[:, i : i + chunk].clone()) for i in range(0, out_channels, chunk)]
        )
        self.biases = torch.nn.ParameterList(
            [torch.nn.Parameter(bias[i : i + chunk].clone()) for i in range(0, out_channels, chunk)]
            if bias is not None
            else []
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        biases = self.biases if len(self.biases) else [None] * len(self.weights)
        output = torch.cat(
            [
                F.conv_transpose1d(x, weight, bias, self.stride, self.padding, 0, 1, self.dilation)
                for weight, bias in zip(self.weights, biases, strict=True)
            ],
            dim=1,
        )
        return output[:, :, self.output_padding :] if self.output_padding else output


class RBLNDAC(RBLNModel):
    """
    RBLN implementation of DAC for text-to-audio models like MOSS-SoundEffect.

    This model is used to accelerate the DAC audio VAE of MOSS-SoundEffect on RBLN NPUs.
    Only the decoder part is compiled, for latent-to-audio conversion.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = DAC
    # The base class looks the HF class up by name in `diffusers`, which has no vendored class.
    _hf_class = DAC
    _rbln_config_class = RBLNDACConfig

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        self.decoder = RBLNRuntimeVAEDecoder(runtime=self.model[0], main_input_name="z")

    @property
    def hop_length(self) -> int:
        return math.prod(self.config.encoder_rates)

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNDACConfig) -> torch.nn.Module:
        # The weight-normed convolutions recompute their weight on every call; fold it once before tracing.
        for module in model.modules():
            if hasattr(module, "weight_g"):
                torch.nn.utils.remove_weight_norm(module)
        for name, module in list(model.decoder.named_modules()):
            if isinstance(module, torch.nn.ConvTranspose1d):
                parent_name, _, child = name.rpartition(".")
                setattr(model.decoder.get_submodule(parent_name), child, RBLNConvTranspose1d(module))
        return _VAEDACDecoder(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        if rbln_config.vae.num_latent_frames is None:
            seconds = rbln_config.vae.max_inference_seconds or pipe.config.max_inference_seconds
            rbln_config.vae.num_latent_frames = pipe.config.sample_rate * seconds // pipe.vae.hop_length
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNDACConfig,
    ) -> RBLNDACConfig:
        if rbln_config.num_latent_frames is None:
            raise ValueError("num_latent_frames must be specified to compile RBLNDAC.")
        input_info = [
            ("z", [rbln_config.batch_size, model_config.latent_dim, rbln_config.num_latent_frames], rbln_config.dtype)
        ]

        compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([compile_config])
        return rbln_config

    def encode(self, audio_data: torch.Tensor, **kwargs: Any):
        raise NotImplementedError("Only the decoder of the DAC is compiled; the pipeline never encodes audio.")

    def decode(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Decode a latent representation into a waveform.

        Args:
            z: The latent representation to decode, shaped (batch, latent_dim, frames).
            kwargs: Ignored.

        Returns:
            The decoded waveform, shaped (batch, 1, frames * hop_length).
        """
        return self.decoder.decode(z.to(self.rbln_config.dtype))


__all__ = [
    "RBLNDAC",
]
