# Copyright 2024 Rebellions Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, List, Union

import rebel
import torch
from diffusers import VQModel
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.autoencoders.vq_model import VQEncoderOutput

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...configurations.models.configuration_vq_model import RBLNVQModelConfig
from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig
from .vae import RBLNRuntimeVQDecoder, RBLNRuntimeVQEncoder, _VQDecoder, _VQEncoder


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

logger = get_logger(__name__)


class RBLNVQModel(RBLNModel):
    """
    RBLN implementation of VQModel for diffusion models.

    This model is used to accelerate VQModel models from diffusers library on RBLN NPUs.
    It can be configured to include both encoder and decoder, or just the decoder part for latent-to-image
    conversion.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    auto_model_class = VQModel
    config_name = "config.json"
    hf_library_name = "diffusers"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        if self.rbln_config.uses_encoder:
            self.encoder = RBLNRuntimeVQEncoder(runtime=self.model[0], main_input_name="x")
        else:
            self.encoder = None

        self.decoder = RBLNRuntimeVQDecoder(runtime=self.model[-1], main_input_name="z")
        self.decoder.lookup_from_codebook = self.config.lookup_from_codebook
        self.image_size = self.rbln_config.image_size

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNModelConfig):
        if rbln_config.uses_encoder:
            expected_models = ["encoder", "decoder"]
        else:
            expected_models = ["decoder"]

        compiled_models = {}
        for i, model_name in enumerate(expected_models):
            if model_name == "encoder":
                wrapped_model = _VQEncoder(model)
            else:
                wrapped_model = _VQDecoder(model)

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
        cls, pipe: RBLNDiffusionMixin, rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNVQModelConfig,
    ) -> RBLNVQModelConfig:
        if hasattr(model_config, "block_out_channels"):
            rbln_config.vqmodel_scale_factor = 2 ** (len(model_config.block_out_channels) - 1)
        else:
            # image processor default value 8 (int)
            rbln_config.vqmodel_scale_factor = 8

        compile_cfgs = []
        if rbln_config.uses_encoder:
            enc_input_info = [
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
            enc_rbln_compile_config = RBLNCompileConfig(compiled_model_name="encoder", input_info=enc_input_info)
            compile_cfgs.append(enc_rbln_compile_config)

        dec_input_info = [
            (
                "h",
                [
                    rbln_config.batch_size,
                    model_config.latent_channels,
                    rbln_config.latent_sample_size[0],
                    rbln_config.latent_sample_size[1],
                ],
                "float32",
            )
        ]
        dec_rbln_compile_config = RBLNCompileConfig(compiled_model_name="decoder", input_info=dec_input_info)
        compile_cfgs.append(dec_rbln_compile_config)

        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNVQModelConfig,
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
        self, x: torch.FloatTensor, return_dict: bool = True, **kwargs: Any
    ) -> Union[torch.FloatTensor, VQEncoderOutput]:
        """
        Encode an input image into a quantized latent representation.

        Args:
            x: The input image to encode.
            return_dict:
                Whether to return output as a dictionary. Defaults to True.
            kwargs: Additional arguments to pass to the encoder/quantizer.

        Returns:
            The quantized latent representation or a specific output object.
        """
        posterior = self.encoder.encode(x)
        if not return_dict:
            return (posterior,)
        return VQEncoderOutput(latents=posterior)

    def decode(
        self, h: torch.FloatTensor, return_dict: bool = True, **kwargs: Any
    ) -> Union[torch.FloatTensor, DecoderOutput]:
        """
        Decode a quantized latent representation back into an image.

        Args:
            h: The quantized latent representation to decode.
            return_dict:
                Whether to return output as a dictionary. Defaults to True.
            kwargs: Additional arguments to pass to the decoder.

        Returns:
            The decoded image or a DecoderOutput object.
        """
        dec, commit_loss = self.decoder.decode(h, **kwargs)
        if not return_dict:
            return (dec, commit_loss)
        return DecoderOutput(sample=dec, commit_loss=commit_loss)
