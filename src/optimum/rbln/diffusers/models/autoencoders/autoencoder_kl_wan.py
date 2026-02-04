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

# from .vae import RBLNRuntimeWanVAEDecoder, RBLNRuntimeWanVAEEncoder, _VAEWanDecoder, _VAEWanEncoder
from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from ...configurations import RBLNAutoencoderKLWanConfig


if TYPE_CHECKING:
    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

logger = get_logger(__name__)

height = 704
width = 1280
PADDED_FRAME_OF_FIRST = 2
CACHE_SIZE_0 = [
    # [1, 3, 1, height, width], # first cache
    [1, 3, PADDED_FRAME_OF_FIRST, height, width],  # padded first cache
    [1, 96, 2, height, width],
    [1, 96, 2, height, width],
    [1, 96, 2, height, width],
    [1, 96, 2, height, width],
    [1, 96, 2, height // 2, width // 2],
    [1, 192, 2, height // 2, width // 2],
    [1, 192, 2, height // 2, width // 2],
    [1, 192, 2, height // 2, width // 2],
    [1, 192, 1, height // 4, width // 4],
    [1, 192, 2, height // 4, width // 4],
    [1, 384, 2, height // 4, width // 4],
    [1, 384, 2, height // 4, width // 4],
    [1, 384, 2, height // 4, width // 4],
    [1, 384, 1, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
]

CACHE_SIZE_N = [
    [1, 3, 2, height, width],  # first cache
    [1, 96, 2, height, width],
    [1, 96, 2, height, width],
    [1, 96, 2, height, width],
    [1, 96, 2, height, width],
    [1, 96, 2, height // 2, width // 2],
    [1, 192, 2, height // 2, width // 2],
    [1, 192, 2, height // 2, width // 2],
    [1, 192, 2, height // 2, width // 2],
    [1, 192, 1, height // 4, width // 4],
    [1, 192, 2, height // 4, width // 4],
    [1, 384, 2, height // 4, width // 4],
    [1, 384, 2, height // 4, width // 4],
    [1, 384, 2, height // 4, width // 4],
    [1, 384, 1, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
    [1, 384, 2, height // 8, width // 8],
]
""" AutoencoderKLWan encode logic 참고용
def clear_cache(self):
    # Use cached conv counts for decoder and encoder to avoid re-iterating modules each call
    self._conv_num = self._cached_conv_counts["decoder"]
    self._conv_idx = [0]
    self._feat_map = [None] * self._conv_num
    # cache encode
    self._enc_conv_num = self._cached_conv_counts["encoder"]
    self._enc_conv_idx = [0]
    self._enc_feat_map = [None] * self._enc_conv_num

def _encode(self, x: torch.Tensor):
    _, _, num_frame, height, width = x.shape

    self.clear_cache()
    if self.config.patch_size is not None:
        x = patchify(x, patch_size=self.config.patch_size)

    if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
        return self.tiled_encode(x)

    iter_ = 1 + (num_frame - 1) // 4
    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        else:
            out_ = self.encoder(
                x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = torch.cat([out, out_], 2)

    enc = self.quant_conv(out)
    self.clear_cache()
    return enc
"""


class _VAEWanEncoder(torch.nn.Module):
    """Wrapper module for Wan VAE encoder extraction."""

    def __init__(self, vae: AutoencoderKLWan):
        super().__init__()
        self.encoder = vae.encoder
        self.cache_dims = CACHE_SIZE_0
        self.encoder.clear_cache()
        self._enc_feat_map = self.encoder._enc_feat_map
        self._enc_conv_idx = self.encoder._enc_conv_idx

    def forward(self, x, *args) -> torch.Tensor:
        out, feat_cache = self.encoder(x, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)

        # post-process: update rbln cache tensors
        dummy_outs = []
        position = torch.tensor(
            0, dtype=torch.int16
        )  # 0 is dummy value -> first output of next chunk have to slice out this frame
        axis = torch.tensor(2, dtype=torch.int16)
        for cache, feat_cache_item, cache_dim in zip(list(args), feat_cache, self.cache_dims):
            n, c, d, h, w = feat_cache_item.shape
            feat_cache_item = feat_cache_item.reshape(n, c, d, -1)
            if cache_dim[2] == 2:
                feat_cache_item = torch.nn.functional.pad(feat_cache_item, (0, 0, 1, 0))  # pad one frame earlier

            dummy_out = torch.ops.rbln_custom_ops.rbln_cache_update(cache, feat_cache_item, position, axis)
            dummy_outs.append(dummy_out)
            print(cache.shape, feat_cache_item.shape)
        return out, dummy_outs


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
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNAutoencoderKLWanConfig) -> torch.nn.Module:
        decoder_model = _VAEWanDecoder(model)
        decoder_model.eval()

        if rbln_config.uses_encoder:
            encoder_model = _VAEWanEncoder(model)
            encoder_model.eval()
            return encoder_model, decoder_model
        else:
            return decoder_model

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNAutoencoderKLWanConfig) -> Dict[str, rebel.RBLNCompiledModel]:
        import pdb

        pdb.set_trace()
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

        if rbln_config.vae.height is None:
            rbln_config.vae.height = 704
        if rbln_config.vae.width is None:
            rbln_config.vae.width = 1280
        if rbln_config.vae.num_frames is None:
            rbln_config.vae.num_frames = 93

        rbln_config.vae.num_channels_latents = pipe.transformer.config.in_channels - 1
        rbln_config.vae.vae_scale_factor_temporal = pipe.vae_scale_factor_temporal
        rbln_config.vae.vae_scale_factor_spatial = pipe.vae_scale_factor_spatial

        return rbln_config

    def get_enc_compile_cfg(self, context, rbln_config):
        # context = CompileContext(use_weight_sharing=False)
        encoder_0_compile_config = rbln_config.compile_cfgs[0]
        encoder_1_compile_config = rbln_config.compile_cfgs[1]

        enc0_example_inputs = encoder_0_compile_config.get_dummy_inputs(encoder_0_compile_config.input_info, fill=0)

        # Mark encoder_0's static tensors (cache)
        static_tensors = {}
        for (name, _, _), tensor in zip(encoder_0_compile_config.input_info, enc0_example_inputs):
            if "feat_cache" in name:
                static_tensors[name] = tensor
                context.mark_static_address(tensor)

        enc1_example_inputs = encoder_1_compile_config.get_dummy_inputs(
            encoder_1_compile_config.input_info, fill=0, static_tensors=static_tensors
        )
        # Mark encoder_1's static tensors (cache)
        for (name, _, _), tensor in zip(encoder_1_compile_config.input_info, enc1_example_inputs):
            if "feat_cache" in name:
                context.mark_static_address(tensor)

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
            vae_enc_0_input_info = [
                (
                    "x",
                    [
                        batch_size,
                        model_config.in_channels,
                        1,  # encode one slice at a time
                        rbln_config.height,
                        rbln_config.width,
                    ],
                    "float32",
                ),
            ]
            CHUNK_SIZE = 4
            vae_enc_1_input_info = [
                (
                    "x",
                    [
                        batch_size,
                        model_config.in_channels,
                        CHUNK_SIZE,  # encode one slice at a time
                        rbln_config.height,
                        rbln_config.width,
                    ],
                    "float32",
                ),
            ]
            cache_0, cache_1 = cls.get_cache_size(rbln_config.height, rbln_config.width)
            for i, (shape_0, shape_1) in enumerate(zip(cache_0, cache_1)):
                shape_0 = [*shape_0[:3], shape_0[-2] * shape_0[-1]]  # N C D HW # FIXME H,W 도 support 가능?
                shape_1 = [*shape_1[:3], shape_1[-2] * shape_1[-1]]  # N C D HW # FIXME H,W 도 support 가능?
                vae_enc_0_input_info.append((f"feat_cache_{i}", shape_0, "float32"))
                vae_enc_1_input_info.append((f"feat_cache_{i}", shape_1, "float32"))

            compile_cfgs.append(RBLNCompileConfig(compiled_model_name="encoder_0", input_info=vae_enc_0_input_info))
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name="encoder_1", input_info=vae_enc_1_input_info))

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

    @classmethod
    def get_cache_size(cls, height, width):
        PADDED_FRAME_OF_FIRST = 2
        CACHE_SIZE_0 = [
            # [1, 3, 1, height, width], # first cache
            [1, 3, PADDED_FRAME_OF_FIRST, height, width],  # padded first cache
            [1, 96, 2, height, width],
            [1, 96, 2, height, width],
            [1, 96, 2, height, width],
            [1, 96, 2, height, width],
            [1, 96, 2, height // 2, width // 2],
            [1, 192, 2, height // 2, width // 2],
            [1, 192, 2, height // 2, width // 2],
            [1, 192, 2, height // 2, width // 2],
            [1, 192, 1, height // 4, width // 4],
            [1, 192, 2, height // 4, width // 4],
            [1, 384, 2, height // 4, width // 4],
            [1, 384, 2, height // 4, width // 4],
            [1, 384, 2, height // 4, width // 4],
            [1, 384, 1, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
        ]

        CACHE_SIZE_N = [
            [1, 3, 2, height, width],  # first cache
            [1, 96, 2, height, width],
            [1, 96, 2, height, width],
            [1, 96, 2, height, width],
            [1, 96, 2, height, width],
            [1, 96, 2, height // 2, width // 2],
            [1, 192, 2, height // 2, width // 2],
            [1, 192, 2, height // 2, width // 2],
            [1, 192, 2, height // 2, width // 2],
            [1, 192, 1, height // 4, width // 4],
            [1, 192, 2, height // 4, width // 4],
            [1, 384, 2, height // 4, width // 4],
            [1, 384, 2, height // 4, width // 4],
            [1, 384, 2, height // 4, width // 4],
            [1, 384, 1, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
            [1, 384, 2, height // 8, width // 8],
        ]
        return CACHE_SIZE_0, CACHE_SIZE_N
