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
from diffusers.models.autoencoders.autoencoder_kl_cosmos import AutoencoderKLCosmos, CosmosCausalConv3d
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from torch.nn import functional as F
from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...configurations import RBLNAutoencoderKLCosmosConfig
from .vae import RBLNRuntimeCosmosVAEDecoder, RBLNRuntimeCosmosVAEEncoder, _VAECosmosDecoder, _VAECosmosEncoder


if TYPE_CHECKING:
    import torch
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

logger = get_logger(__name__)


class RBLNAutoencoderKLCosmos(RBLNModel):
    """
    RBLN implementation of AutoencoderKLCosmos for diffusion models.

    This model is used to accelerate AutoencoderKLCosmos models from diffusers library on RBLN NPUs.
    It can be configured to include both encoder and decoder, or just the decoder part for latent-to-video
    conversion.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    auto_model_class = AutoencoderKLCosmos
    hf_library_name = "diffusers"
    _rbln_config_class = RBLNAutoencoderKLCosmosConfig

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        if self.rbln_config.uses_encoder:
            self.encoder = RBLNRuntimeCosmosVAEEncoder(
                runtime=self.model[0], main_input_name="x", use_slicing=self.rbln_config.use_slicing
            )

        self.decoder = RBLNRuntimeCosmosVAEDecoder(
            runtime=self.model[-1], main_input_name="z", use_slicing=self.rbln_config.use_slicing
        )
        self.image_size = self.rbln_config.image_size

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNAutoencoderKLCosmosConfig
    ) -> torch.nn.Module:
        decoder_model = _VAECosmosDecoder(model)
        decoder_model.eval()

        if rbln_config.uses_encoder:
            encoder_model = _VAECosmosEncoder(model)
            encoder_model.eval()
            return encoder_model, decoder_model
        else:
            return decoder_model

    @classmethod
    def get_compiled_model(
        cls, model, rbln_config: RBLNAutoencoderKLCosmosConfig
    ) -> Dict[str, rebel.RBLNCompiledModel]:
        def replaced_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            if self.temporal_pad != 0:
                hidden_states_prev = hidden_states[:, :, :1, ...].repeat(1, 1, self.temporal_pad, 1, 1)
                hidden_states = torch.cat([hidden_states_prev, hidden_states], dim=2)
            hidden_states = F.pad(hidden_states, (*self.spatial_pad, 0, 0), mode=self.pad_mode, value=0.0)
            return super(CosmosCausalConv3d, self).forward(hidden_states)

        try:
            original_forward = CosmosCausalConv3d.forward
            CosmosCausalConv3d.forward = replaced_forward

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

        finally:
            CosmosCausalConv3d.forward = original_forward

        return compiled_models

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        rbln_config.vae.num_channels_latents = pipe.transformer.config.out_channels
        rbln_config.vae.vae_scale_factor_temporal = pipe.vae_scale_factor_temporal
        rbln_config.vae.vae_scale_factor_spatial = pipe.vae_scale_factor_spatial
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNAutoencoderKLCosmosConfig,
    ) -> RBLNAutoencoderKLCosmosConfig:
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
        rbln_config: RBLNAutoencoderKLCosmosConfig,
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
        self, x: torch.FloatTensor, return_dict: bool = True, **kwargs: Dict[str, Any]
    ) -> Union[torch.FloatTensor, AutoencoderKLOutput]:
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

    def decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[torch.FloatTensor, DecoderOutput]:
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
