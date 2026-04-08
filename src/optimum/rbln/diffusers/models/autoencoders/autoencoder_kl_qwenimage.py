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

from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import rebel
import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from ...configurations import RBLNAutoencoderKLQwenImageConfig


if TYPE_CHECKING:
    import torch
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

logger = get_logger(__name__)


class _VAEQwenImageEncoder(torch.nn.Module):
    """Wrapper to trace the encoder path of AutoencoderKLQwenImage."""

    def __init__(self, vae: "AutoencoderKLQwenImage"):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, channels, frames, height, width]
        # AutoencoderKLQwenImage._encode returns the raw encoded tensor (before DiagonalGaussianDistribution)
        h = self.vae._encode(x)
        return h


class _VAEQwenImageDecoder(torch.nn.Module):
    """Wrapper to trace the decoder path of AutoencoderKLQwenImage."""

    def __init__(self, vae: "AutoencoderKLQwenImage"):
        super().__init__()
        self.vae = vae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: [batch, z_dim, frames, height, width]
        dec = self.vae._decode(z, return_dict=False)
        return dec


class RBLNRuntimeQwenImageVAEEncoder(RBLNPytorchRuntime):
    def encode(self, x: torch.FloatTensor, **kwargs) -> DiagonalGaussianDistribution:
        moments = self.forward(x.contiguous())
        posterior = DiagonalGaussianDistribution(moments)
        return posterior


class RBLNRuntimeQwenImageVAEDecoder(RBLNPytorchRuntime):
    def decode(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.forward(z)


class RBLNAutoencoderKLQwenImage(RBLNModel):
    """
    RBLN implementation of AutoencoderKLQwenImage (3D causal VAE) for Qwen-Image pipelines.

    This model accelerates AutoencoderKLQwenImage from diffusers on RBLN NPUs.
    It supports both encoder and decoder paths for image editing workflows that
    require encoding input images and decoding edited latents.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    auto_model_class = AutoencoderKLQwenImage
    hf_library_name = "diffusers"
    _rbln_config_class = RBLNAutoencoderKLQwenImageConfig

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        if self.rbln_config.uses_encoder:
            self.encoder = RBLNRuntimeQwenImageVAEEncoder(runtime=self.model[0], main_input_name="x")
        else:
            self.encoder = None

        self.decoder = RBLNRuntimeQwenImageVAEDecoder(runtime=self.model[-1], main_input_name="z")
        self.image_size = self.rbln_config.image_size

    @classmethod
    def get_compiled_model(
        cls, model, rbln_config: RBLNAutoencoderKLQwenImageConfig
    ) -> Dict[str, rebel.RBLNCompiledModel]:
        if rbln_config.uses_encoder:
            expected_models = ["encoder", "decoder"]
        else:
            expected_models = ["decoder"]

        compiled_models = {}
        for i, model_name in enumerate(expected_models):
            if model_name == "encoder":
                wrapped_model = _VAEQwenImageEncoder(model)
            else:
                wrapped_model = _VAEQwenImageDecoder(model)

            wrapped_model.eval()

            compiled_models[model_name] = cls.compile(
                wrapped_model,
                rbln_compile_config=rbln_config.compile_cfgs[i],
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device_map[model_name],
            )

        return compiled_models

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        vae_config = rbln_config.vae

        if vae_config.sample_size is None:
            transformer = getattr(pipe, "transformer", None)
            if transformer is not None and hasattr(transformer.config, "sample_size"):
                sample_size = transformer.config.sample_size
                if isinstance(sample_size, int):
                    sample_size = (sample_size, sample_size)
                vae_scale_factor = pipe.vae_scale_factor
                vae_config.sample_size = (
                    sample_size[0] * vae_scale_factor,
                    sample_size[1] * vae_scale_factor,
                )

        if vae_config.vae_scale_factor is None:
            vae_config.vae_scale_factor = pipe.vae_scale_factor

        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNAutoencoderKLQwenImageConfig,
    ) -> RBLNAutoencoderKLQwenImageConfig:
        if isinstance(rbln_config.sample_size, int):
            rbln_config.sample_size = (rbln_config.sample_size, rbln_config.sample_size)

        if rbln_config.z_dim is None:
            rbln_config.z_dim = model_config.z_dim

        if rbln_config.input_channels is None:
            rbln_config.input_channels = model_config.input_channels

        if rbln_config.vae_scale_factor is None:
            rbln_config.vae_scale_factor = 2 ** len(model_config.temperal_downsample)

        compile_cfgs = []

        # For image editing, frames=1 (single image)
        num_frames = 1

        if rbln_config.uses_encoder:
            vae_enc_input_info = [
                (
                    "x",
                    [
                        rbln_config.batch_size,
                        rbln_config.input_channels,
                        num_frames,
                        rbln_config.sample_size[0],
                        rbln_config.sample_size[1],
                    ],
                    "float32",
                )
            ]
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name="encoder", input_info=vae_enc_input_info))

        latent_height = rbln_config.sample_size[0] // rbln_config.vae_scale_factor
        latent_width = rbln_config.sample_size[1] // rbln_config.vae_scale_factor

        vae_dec_input_info = [
            (
                "z",
                [
                    rbln_config.batch_size,
                    rbln_config.z_dim,
                    num_frames,
                    latent_height,
                    latent_width,
                ],
                "float32",
            )
        ]
        compile_cfgs.append(RBLNCompileConfig(compiled_model_name="decoder", input_info=vae_dec_input_info))

        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNAutoencoderKLQwenImageConfig,
    ) -> List[rebel.Runtime]:
        if len(compiled_models) == 1:
            expected_models = ["decoder"]
        else:
            expected_models = ["encoder", "decoder"]

        if any(model_name not in rbln_config.device_map for model_name in expected_models):
            cls._raise_missing_compiled_file_error(expected_models)

        device_vals = [rbln_config.device_map[model_name] for model_name in expected_models]
        return [
            rebel.Runtime(
                compiled_model,
                tensor_type="pt",
                device=device_val,
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for compiled_model, device_val in zip(compiled_models, device_vals, strict=False)
        ]

    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True, **kwargs: Dict[str, Any]
    ) -> Union[torch.FloatTensor, AutoencoderKLOutput]:
        """
        Encode an input image into a latent representation.

        Args:
            x: Input image tensor of shape [batch, channels, frames, height, width].
            return_dict: Whether to return output as AutoencoderKLOutput. Defaults to True.

        Returns:
            The latent representation or AutoencoderKLOutput if return_dict=True
        """
        posterior = self.encoder.encode(x)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, **kwargs: Dict[str, Any]
    ) -> Union[torch.FloatTensor, DecoderOutput]:
        """
        Decode a latent representation into an image.

        Args:
            z: Latent tensor of shape [batch, z_dim, frames, height, width].
            return_dict: Whether to return output as DecoderOutput. Defaults to True.

        Returns:
            The decoded image or DecoderOutput if return_dict=True
        """
        dec = self.decoder.decode(z)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
