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

from typing import TYPE_CHECKING, Any, Dict, List, Union

import rebel
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from ...configurations import RBLNAutoencoderKLWanConfig
from .vae import RBLNRuntimeWanVAEDecoder, RBLNRuntimeWanVAEEncoder, _VAEWanDecoder, _VAEWanEncoder


if TYPE_CHECKING:
    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

logger = get_logger(__name__)


class _VAEWanEncoder(torch.nn.Module):
    """Wrapper module for Wan VAE encoder extraction."""

    def __init__(self, vae: AutoencoderKLWan):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae._encode(x)


class _VAEWanDecoder(torch.nn.Module):
    """Wrapper module for Wan VAE decoder extraction."""

    def __init__(self, vae: AutoencoderKLWan):
        super().__init__()
        self.vae = vae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae._decode(z, return_dict=False)[0]


class RBLNRuntimeWanVAEEncoder(RBLNPytorchRuntime):
    """Runtime wrapper for Wan VAE encoder inference."""

    def encode(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.forward(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.forward(x)
        posterior = DiagonalGaussianDistribution(h)
        return posterior


class RBLNRuntimeWanVAEDecoder(RBLNPytorchRuntime):
    """Runtime wrapper for Wan VAE decoder inference."""

    def decode(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self.forward(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self.forward(z)
        return decoded


class RBLNAutoencoderKLWan(RBLNModel):
    """
    RBLN implementation of AutoencoderKLWan for diffusion models.

    This model is used to accelerate AutoencoderKLWan models from diffusers library on RBLN NPUs.
    It can be configured to include both encoder and decoder, or just the decoder part for latent-to-video
    conversion.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    auto_model_class = AutoencoderKLWan
    hf_library_name = "diffusers"
    _rbln_config_class = RBLNAutoencoderKLWanConfig

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.temperal_downsample = self.config.temperal_downsample

        if self.rbln_config.uses_encoder:
            self.encoder = RBLNRuntimeWanVAEEncoder(
                runtime=self.model[0], main_input_name="x", use_slicing=self.rbln_config.use_slicing
            )

        self.decoder = RBLNRuntimeWanVAEDecoder(
            runtime=self.model[-1], main_input_name="z", use_slicing=self.rbln_config.use_slicing
        )
        self.image_size = self.rbln_config.image_size

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNAutoencoderKLWanConfig
    ) -> torch.nn.Module:
        decoder_model = _VAEWanDecoder(model)
        decoder_model.eval()

        if rbln_config.uses_encoder:
            encoder_model = _VAEWanEncoder(model)
            encoder_model.eval()
            return encoder_model, decoder_model
        else:
            return decoder_model

    @classmethod
    def get_compiled_model(
        cls, model, rbln_config: RBLNAutoencoderKLWanConfig
    ) -> Dict[str, rebel.RBLNCompiledModel]:
        compiled_models = {}
        if rbln_config.uses_encoder:
            encoder_model, decoder_model = cls._wrap_model_if_needed(model, rbln_config)
            enc_compiled_model = cls.compile(
                encoder_model,
                rbln_compile_config=rbln_config.compile_cfgs[0],
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device_map["encoder"],
            )
            compiled_models["encoder"] = enc_compiled_model
        else:
            decoder_model = cls._wrap_model_if_needed(model, rbln_config)

        dec_compiled_model = cls.compile(
            decoder_model,
            rbln_compile_config=rbln_config.compile_cfgs[-1],
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device_map["decoder"],
        )
        compiled_models["decoder"] = dec_compiled_model

        return compiled_models

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        # For Cosmos2.5 pipeline, get latent channels from transformer config
        # transformer.config.in_channels - 1 is the num_channels_latents (minus 1 for condition mask)
        # rbln_config.vae.num_channels_latents = pipe.transformer.config.in_channels - 1
        rbln_config.vae.num_channels_latents = 93
        rbln_config.vae.vae_scale_factor_temporal = pipe.vae_scale_factor_temporal
        rbln_config.vae.vae_scale_factor_spatial = pipe.vae_scale_factor_spatial
        
        # if rbln_config.vae.height is None:
        #     rbln_config.vae.height = 704 if rbln_config.vae.uses_encoder else 768
        # if rbln_config.vae.width is None:
        #     rbln_config.vae.width = 1280 if rbln_config.vae.uses_encoder else 1360

        # rbln_config.vae.num_channels_latents = pipe.transformer.config.in_channels - int(rbln_config.vae.uses_encoder)
        # rbln_config.vae.vae_scale_factor_temporal = pipe.vae_scale_factor_temporal
        # rbln_config.vae.vae_scale_factor_spatial = pipe.vae_scale_factor_spatial
        # if rbln_config.vae.num_frames is None:
        #     rbln_config.vae.num_frames = 93 if rbln_config.vae.uses_encoder else 1
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNAutoencoderKLWanConfig,
    ) -> RBLNAutoencoderKLWanConfig:
        batch_size = 1 if rbln_config.use_slicing else rbln_config.batch_size
        compile_cfgs = []
        if rbln_config.uses_encoder:
            vae_enc_input_info = [
                (
                    "x",
                    [
                        batch_size,
                        model_config.in_channels,
                        rbln_config.num_frames,
                        rbln_config.height,
                        rbln_config.width,
                    ],
                    "float32",
                ),
            ]
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name="encoder", input_info=vae_enc_input_info))

        num_latent_frames = (rbln_config.num_frames - 1) // rbln_config.vae_scale_factor_temporal + 1
        latent_height = rbln_config.height // rbln_config.vae_scale_factor_spatial
        latent_width = rbln_config.width // rbln_config.vae_scale_factor_spatial

        vae_dec_input_info = [
            (
                "z",
                [
                    batch_size,
                    rbln_config.num_channels_latents,
                    num_latent_frames,
                    latent_height,
                    latent_width,
                ],
                "float32",
            ),
        ]
        compile_cfgs.append(RBLNCompileConfig(compiled_model_name="decoder", input_info=vae_dec_input_info))

        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNAutoencoderKLWanConfig,
    ) -> List[rebel.Runtime]:
        if len(compiled_models) == 1:
            # decoder
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
            for compiled_model, device_val in zip(compiled_models, device_vals)
        ]

    def encode(
        self, x: torch.Tensor, return_dict: bool = True, **kwargs: Dict[str, Any]
    ) -> Union[torch.Tensor, AutoencoderKLOutput]:
        """
        Encode an input video into a latent representation.

        Args:
            x: The input video to encode.
            return_dict:
                Whether to return output as a dictionary. Defaults to True.
            kwargs: Additional arguments to pass to the encoder.

        Returns:
            The latent representation or AutoencoderKLOutput if return_dict=True
        """
        posterior = self.encoder.encode(x)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[torch.Tensor, DecoderOutput]:
        """
        Decode a latent representation into a video.

        Args:
            z: The latent representation to decode.
            return_dict:
                Whether to return output as a dictionary. Defaults to True.

        Returns:
            The decoded video or DecoderOutput if return_dict=True
        """
        decoded = self.decoder.decode(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
