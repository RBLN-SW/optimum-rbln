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
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...configurations import RBLNAutoencoderKLConfig
from .vae import RBLNRuntimeVAEDecoder, RBLNRuntimeVAEEncoder, _VAEDecoder, _VAEEncoder


if TYPE_CHECKING:
    import torch
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

logger = get_logger(__name__)


class RBLNAutoencoderKL(RBLNModel):
    """
    RBLN implementation of AutoencoderKL (VAE) for diffusion models.

    This model is used to accelerate AutoencoderKL (VAE) models from diffusers library on RBLN NPUs.
    It can be configured to include both encoder and decoder, or just the decoder part for latent-to-image
    conversion.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    auto_model_class = AutoencoderKL
    hf_library_name = "diffusers"
    _rbln_config_class = RBLNAutoencoderKLConfig

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        if self.rbln_config.uses_encoder:
            self.encoder = RBLNRuntimeVAEEncoder(runtime=self.model[0], main_input_name="x")
        else:
            self.encoder = None

        self.decoder = RBLNRuntimeVAEDecoder(runtime=self.model[-1], main_input_name="z")
        self.image_size = self.rbln_config.image_size

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNAutoencoderKLConfig) -> Dict[str, rebel.RBLNCompiledModel]:
        if rbln_config.uses_encoder:
            expected_models = ["encoder", "decoder"]
        else:
            expected_models = ["decoder"]

        compiled_models = {}
        for i, model_name in enumerate(expected_models):
            if model_name == "encoder":
                wrapped_model = _VAEEncoder(model)
            else:
                wrapped_model = _VAEDecoder(model)

            wrapped_model.eval()

            compiled_models[model_name] = cls.compile(
                wrapped_model,
                rbln_compile_config=rbln_config.compile_cfgs[i],
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device_map[model_name],
            )

        return compiled_models

    @classmethod
    def get_vae_sample_size(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: RBLNAutoencoderKLConfig, return_vae_scale_factor: bool = False
    ) -> Tuple[int, int]:
        sample_size = rbln_config.sample_size
        noise_module = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
        vae_scale_factor = (
            pipe.vae_scale_factor
            if hasattr(pipe, "vae_scale_factor")
            else 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        )

        if noise_module is None:
            raise AttributeError(
                "Cannot find noise processing or predicting module attributes. ex. U-Net, Transformer, ..."
            )

        if sample_size is None:
            sample_size = noise_module.config.sample_size
            if isinstance(sample_size, int):
                sample_size = (sample_size, sample_size)
            sample_size = (sample_size[0] * vae_scale_factor, sample_size[1] * vae_scale_factor)

        if return_vae_scale_factor:
            return sample_size, vae_scale_factor
        else:
            return sample_size

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        rbln_config.vae.sample_size, rbln_config.vae.vae_scale_factor = cls.get_vae_sample_size(
            pipe, rbln_config.vae, return_vae_scale_factor=True
        )
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNAutoencoderKLConfig,
    ) -> RBLNAutoencoderKLConfig:
        if rbln_config.sample_size is None:
            rbln_config.sample_size = model_config.sample_size

        if isinstance(rbln_config.sample_size, int):
            rbln_config.sample_size = (rbln_config.sample_size, rbln_config.sample_size)

        if rbln_config.in_channels is None:
            rbln_config.in_channels = model_config.in_channels

        if rbln_config.latent_channels is None:
            rbln_config.latent_channels = model_config.latent_channels

        if rbln_config.vae_scale_factor is None:
            if hasattr(model_config, "block_out_channels"):
                rbln_config.vae_scale_factor = 2 ** (len(model_config.block_out_channels) - 1)
            else:
                # vae image processor default value 8 (int)
                rbln_config.vae_scale_factor = 8

        compile_cfgs = []
        if rbln_config.uses_encoder:
            vae_enc_input_info = [
                (
                    "x",
                    [
                        rbln_config.batch_size,
                        rbln_config.in_channels,
                        rbln_config.sample_size[0],
                        rbln_config.sample_size[1],
                    ],
                    "float32",
                )
            ]
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name="encoder", input_info=vae_enc_input_info))

        vae_dec_input_info = [
            (
                "z",
                [
                    rbln_config.batch_size,
                    rbln_config.latent_channels,
                    rbln_config.latent_sample_size[0],
                    rbln_config.latent_sample_size[1],
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
        rbln_config: RBLNAutoencoderKLConfig,
    ) -> List[rebel.Runtime]:
        if len(compiled_models) == 1:
            # decoder
            expected_models = ["decoder"]
        else:
            # encoder, decoder
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
        self, x: torch.FloatTensor, return_dict: bool = True, **kwargs: Dict[str, Any]
    ) -> Union[torch.FloatTensor, AutoencoderKLOutput]:
        """
        Encode an input image into a latent representation.

        Args:
            x: The input image to encode.
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

    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, **kwargs: Dict[str, Any]
    ) -> Union[torch.FloatTensor, DecoderOutput]:
        """
        Decode a latent representation into an image.

        Args:
            z: The latent representation to decode.
            return_dict:
                Whether to return output as a dictionary. Defaults to True.
            kwargs: Additional arguments to pass to the decoder.

        Returns:
            The decoded image or DecoderOutput if return_dict=True
        """
        dec = self.decoder.decode(z)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
