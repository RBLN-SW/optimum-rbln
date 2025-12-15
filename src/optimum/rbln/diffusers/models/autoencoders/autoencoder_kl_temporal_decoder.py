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

from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import rebel
import torch  # noqa: I001
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...configurations import RBLNAutoencoderKLTemporalDecoderConfig
from ...modeling_diffusers import RBLNDiffusionMixin
from .vae import (
    DiagonalGaussianDistribution,
    RBLNRuntimeVAEDecoder,
    RBLNRuntimeVAEEncoder,
    _VAEEncoder,
    _VAETemporalDecoder,
)


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

logger = get_logger(__name__)


class RBLNAutoencoderKLTemporalDecoder(RBLNModel):
    auto_model_class = AutoencoderKLTemporalDecoder
    hf_library_name = "diffusers"
    _rbln_config_class = RBLNAutoencoderKLTemporalDecoderConfig

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        if self.rbln_config.uses_encoder:
            self.encoder = RBLNRuntimeVAEEncoder(runtime=self.model[0], main_input_name="x")
        self.decoder = RBLNRuntimeVAEDecoder(runtime=self.model[-1], main_input_name="z")
        self.image_size = self.rbln_config.image_size

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNAutoencoderKLTemporalDecoderConfig
    ) -> torch.nn.Module:
        decoder_model = _VAETemporalDecoder(model)
        decoder_model.num_frames = rbln_config.decode_chunk_size
        decoder_model.eval()

        if rbln_config.uses_encoder:
            encoder_model = _VAEEncoder(model)
            encoder_model.eval()
            return encoder_model, decoder_model
        else:
            return decoder_model

    @classmethod
    def get_compiled_model(
        cls, model, rbln_config: RBLNAutoencoderKLTemporalDecoderConfig
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
    def get_vae_sample_size(
        cls,
        pipe: "RBLNDiffusionMixin",
        rbln_config: RBLNAutoencoderKLTemporalDecoderConfig,
        return_vae_scale_factor: bool = False,
    ) -> Tuple[int, int]:
        sample_size = rbln_config.sample_size
        if hasattr(pipe, "vae_scale_factor"):
            vae_scale_factor = pipe.vae_scale_factor
        else:
            if hasattr(pipe.vae.config, "block_out_channels"):
                vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
            else:
                vae_scale_factor = 8  # vae image processor default value 8 (int)

        if sample_size is None:
            sample_size = pipe.unet.config.sample_size
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

        if rbln_config.vae.num_frames is None:
            if hasattr(pipe.unet.config, "num_frames"):
                rbln_config.vae.num_frames = pipe.unet.config.num_frames
            else:
                raise ValueError("num_frames should be specified in unet config.json")

        if rbln_config.vae.decode_chunk_size is None:
            rbln_config.vae.decode_chunk_size = rbln_config.vae.num_frames

        def chunk_frame(num_frames, decode_chunk_size):
            # get closest divisor to num_frames
            divisors = [i for i in range(1, num_frames) if num_frames % i == 0]
            closest = min(divisors, key=lambda x: abs(x - decode_chunk_size))
            if decode_chunk_size != closest:
                logger.warning(
                    f"To ensure successful model compilation and prevent device OOM, {decode_chunk_size} is set to {closest}."
                )
            return closest

        decode_chunk_size = chunk_frame(rbln_config.vae.num_frames, rbln_config.vae.decode_chunk_size)
        rbln_config.vae.decode_chunk_size = decode_chunk_size
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNAutoencoderKLTemporalDecoderConfig,
    ) -> RBLNAutoencoderKLTemporalDecoderConfig:
        if rbln_config.sample_size is None:
            rbln_config.sample_size = model_config.sample_size

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
                        model_config.in_channels,
                        rbln_config.sample_size[0],
                        rbln_config.sample_size[1],
                    ],
                    "float32",
                )
            ]
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name="encoder", input_info=vae_enc_input_info))

        decode_batch_size = rbln_config.batch_size * rbln_config.decode_chunk_size
        vae_dec_input_info = [
            (
                "z",
                [
                    decode_batch_size,
                    model_config.latent_channels,
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
        rbln_config: RBLNAutoencoderKLTemporalDecoderConfig,
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
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode an input image into a latent representation.

        Args:
            x: The input image to encode.
            return_dict:
                Whether to return output as a dictionary. Defaults to True.

        Returns:
            The latent representation or AutoencoderKLOutput if return_dict=True
        """
        posterior = self.encoder.encode(x)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.FloatTensor, return_dict: bool = True) -> torch.FloatTensor:
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
