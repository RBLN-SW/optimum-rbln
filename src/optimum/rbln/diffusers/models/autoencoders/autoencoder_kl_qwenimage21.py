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

import rebel
import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from rebel.compile_context import CompileContext

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ....utils.logging import get_logger
from ...configurations import RBLNAutoencoderKLQwenImage21Config
from .vae import (
    RBLNRuntimeQwenImage21VAEDecoder,
    RBLNRuntimeQwenImage21VAEEncoder,
    _VAEQwenImage21Decoder,
    _VAEQwenImage21Encoder,
)


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


logger = get_logger(__name__)


def _get_graph_name(kind: str, height: int, width: int) -> str:
    return f"{kind}_{height}x{width}"


class RBLNAutoencoderKLQwenImage21(RBLNModel):
    """
    RBLN implementation of AutoencoderKLQwenImage21 for diffusion models.

    This model is used to accelerate AutoencoderKLQwenImage21 models from diffusers library on RBLN NPUs.
    It can be configured to include both encoder and decoder, or just the decoder part for latent-to-image
    conversion. The decoder compiles one graph for every output size and the encoder one for every canvas size.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    auto_model_class = AutoencoderKLQwenImage21
    hf_library_name = "diffusers"
    _rbln_config_class = RBLNAutoencoderKLQwenImage21Config

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        # Indexed, not zipped: without runtimes, `self.model` is one `UnavailableRuntime` for every index.
        runtimes = {cfg.compiled_model_name: self.model[i] for i, cfg in enumerate(self.rbln_config.compile_cfgs)}
        scale = self.config.scale_factor_spatial

        if self.rbln_config.uses_encoder:
            canvases = {
                (height, width): runtimes[_get_graph_name("encoder", height, width)]
                for height, width in self.rbln_config.encoder_sample_size
            }
            self.encoder = RBLNRuntimeQwenImage21VAEEncoder(
                runtime=self.model[0],
                main_input_name="x",
                canvases=canvases,
                scale_factor=scale,
                dtype=self.rbln_config.dtype,
            )

        # Keyed by the latent (height, width) the decoder takes.
        decoders = {
            (height // scale, width // scale): runtimes[_get_graph_name("decoder", height, width)]
            for height, width in self.rbln_config.sample_size
        }
        self.decoder = RBLNRuntimeQwenImage21VAEDecoder(
            runtime=self.model[-1], main_input_name="z", decoders=decoders, scale_factor=scale
        )
        self.image_size = self.rbln_config.image_size

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNAutoencoderKLQwenImage21Config
    ) -> dict[str, torch.nn.Module]:
        encoder, decoder = _VAEQwenImage21Encoder(model).eval(), _VAEQwenImage21Decoder(model).eval()
        return {
            cfg.compiled_model_name: encoder if cfg.compiled_model_name.startswith("encoder") else decoder
            for cfg in rbln_config.compile_cfgs
        }

    @classmethod
    def get_compiled_model(
        cls, model: "PreTrainedModel", rbln_config: RBLNAutoencoderKLQwenImage21Config
    ) -> dict[str, rebel.RBLNCompiledModel]:
        wrapped_models = cls._wrap_model_if_needed(model, rbln_config)
        # Graphs of one kind wrap the same encoder or decoder: share its weights across the image sizes.
        contexts = {
            "encoder": CompileContext(use_weight_sharing=True),
            "decoder": CompileContext(use_weight_sharing=True),
        }
        compiled_models = {}
        for compile_config in rbln_config.compile_cfgs:
            name = compile_config.compiled_model_name
            compiled_models[name] = cls.compile(
                wrapped_models[name],
                compile_config,
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device_map[name],
                compile_context=contexts[name.split("_")[0]],
            )
        return compiled_models

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNAutoencoderKLQwenImage21Config,
    ) -> RBLNAutoencoderKLQwenImage21Config:
        if rbln_config.sample_size is None:
            raise ValueError(
                "`sample_size` must be specified to compile the Qwen-Image 2.1 VAE. Compiling through a pipeline "
                "fills it with that pipeline's defaults; a standalone compile must pass it in rbln_config."
            )
        if rbln_config.uses_encoder is None:
            rbln_config.uses_encoder = False
        if rbln_config.uses_encoder and rbln_config.encoder_sample_size is None:
            rbln_config.encoder_sample_size = rbln_config.sample_size

        scale = model_config.scale_factor_spatial
        sizes = list(rbln_config.sample_size) + list(rbln_config.encoder_sample_size or [])
        not_divisible = [size for size in sizes if size[0] % scale or size[1] % scale]
        if not_divisible:
            raise ValueError(f"Image sizes {not_divisible} must be multiples of the VAE scale factor ({scale}).")

        batch_size = rbln_config.batch_size
        dtype = rbln_config.dtype
        compile_cfgs = []
        if rbln_config.uses_encoder:
            for height, width in rbln_config.encoder_sample_size:
                input_info = [
                    ("x", [batch_size, model_config.in_channels, 1, height, width], dtype),
                    ("mask", [1, 1, 1, height, width], dtype),
                ]
                compile_cfgs.append(
                    RBLNCompileConfig(
                        compiled_model_name=_get_graph_name("encoder", height, width), input_info=input_info
                    )
                )
        for height, width in rbln_config.sample_size:
            input_info = [("z", [batch_size, model_config.z_dim, 1, height // scale, width // scale], dtype)]
            compile_cfgs.append(
                RBLNCompileConfig(compiled_model_name=_get_graph_name("decoder", height, width), input_info=input_info)
            )
        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: list[rebel.RBLNCompiledModel],
        rbln_config: RBLNAutoencoderKLQwenImage21Config,
    ) -> list[rebel.Runtime]:
        names = [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs]
        if any(name not in rbln_config.device_map for name in names):
            cls._raise_missing_compiled_file_error(names)

        return [
            rebel.Runtime(
                compiled_model,
                tensor_type="pt",
                device=rbln_config.device_map[name],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for name, compiled_model in zip(names, compiled_models, strict=True)
        ]

    def encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput | tuple:
        """
        Encode an input image into a latent representation.

        Args:
            x: The input image to encode, shaped (batch_size, channels, 1, height, width). It runs at the top left
                of the smallest compiled canvas that holds it.
            return_dict:
                Whether to return output as a dictionary. Defaults to True.

        Returns:
            The latent representation or AutoencoderKLOutput if return_dict=True
        """
        posterior = self.encoder.encode(x.to(self.rbln_config.dtype))
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple:
        """
        Decode a latent representation into an image.

        Args:
            z: The latent representation to decode, shaped (batch_size, z_dim, 1, latent_height, latent_width).
            return_dict:
                Whether to return output as a dictionary. Defaults to True.

        Returns:
            The decoded image or DecoderOutput if return_dict=True
        """
        decoded = self.decoder.decode(z.to(self.rbln_config.dtype))
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)


__all__ = [
    "RBLNAutoencoderKLQwenImage21",
]
