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

from typing import TYPE_CHECKING, Any, Dict, List, Union, Tuple, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan, patchify, unpatchify
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

def get_cache_size_enc(height=704, width=1280):
    # 사실상 처음에는 전부 1만 나오는데, 그 다음부터 cache frame이 2개씩 쌓이므로 미리 패딩해서 넣어놓음
    CACHE_SIZE_0 = [
        # [1, 3, 1, height, width], # first cache
        [1, 3, 1, height, width],  # padded first cache
        [1, 96, 1, height, width],
        [1, 96, 1, height, width],
        [1, 96, 1, height, width],
        [1, 96, 1, height, width],
        [1, 96, 1, height // 2, width // 2],
        [1, 192, 1, height // 2, width // 2],
        [1, 192, 1, height // 2, width // 2],
        [1, 192, 1, height // 2, width // 2],
        [1, 192, 1, height // 4, width // 4],
        [1, 192, 1, height // 4, width // 4],
        [1, 384, 1, height // 4, width // 4],
        [1, 384, 1, height // 4, width // 4],
        [1, 384, 1, height // 4, width // 4],
        [1, 384, 1, height // 8, width // 8],
        [1, 384, 1, height // 8, width // 8],
        [1, 384, 1, height // 8, width // 8],
        [1, 384, 1, height // 8, width // 8],
        [1, 384, 1, height // 8, width // 8],
        [1, 384, 1, height // 8, width // 8],
        [1, 384, 1, height // 8, width // 8],
        [1, 384, 1, height // 8, width // 8],
        [1, 384, 1, height // 8, width // 8],
        [1, 384, 1, height // 8, width // 8],
    ]
    PADDED_FRAME = 2
    NO_PAD_INDICES = {9, 14}  # pre-defined indices where frame dim is always 1
    PADDED_CACHE_SIZE_0 = [
        [s[0], s[1], s[2] if i in NO_PAD_INDICES else PADDED_FRAME, s[3], s[4]]
        for i, s in enumerate(CACHE_SIZE_0)
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
    # 두개 다 같은 거 아닌가?
    return PADDED_CACHE_SIZE_0, CACHE_SIZE_N

def get_cache_size_dec(height=704, width=1280):
    CACHE_SIZE_0 = [        
        [1, 16, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 192, 2, height // 4, width // 4],
        [1, 384, 2, height // 4, width // 4],
        [1, 384, 2, height // 4, width // 4],
        [1, 384, 2, height // 4, width // 4],
        [1, 384, 2, height // 4, width // 4],
        [1, 384, 2, height // 4, width // 4],
        [1, 384, 2, height // 4, width // 4],
        [1, 192, 2, height // 2, width // 2],
        [1, 192, 2, height // 2, width // 2],
        [1, 192, 2, height // 2, width // 2],
        [1, 192, 2, height // 2, width // 2],
        [1, 192, 2, height // 2, width // 2],
        [1, 192, 2, height // 2, width // 2],
        [1, 96, 2, height, width],
        [1, 96, 2, height, width],
        [1, 96, 2, height, width],
        [1, 96, 2, height, width],
        [1, 96, 2, height, width],
        [1, 96, 2, height, width],
        [1, 96, 2, height, width],
    ]

    CACHE_SIZE_N = [
        [1, 16, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 384, 2, height // 8, width // 8],
        [1, 192, 2, height // 4, width // 4],
        [1, 384, 2, height // 4, width // 4],
        [1, 384, 2, height // 4, width // 4],
        [1, 384, 2, height // 4, width // 4],
        [1, 384, 2, height // 4, width // 4],
        [1, 384, 2, height // 4, width // 4],
        [1, 384, 2, height // 4, width // 4],
        [1, 192, 2, height // 2, width // 2],
        [1, 192, 2, height // 2, width // 2],
        [1, 192, 2, height // 2, width // 2],
        [1, 192, 2, height // 2, width // 2],
        [1, 192, 2, height // 2, width // 2],
        [1, 192, 2, height // 2, width // 2],
        [1, 96, 2, height, width],
        [1, 96, 2, height, width],
        [1, 96, 2, height, width],
        [1, 96, 2, height, width],
        [1, 96, 2, height, width],
        [1, 96, 2, height, width],
        [1, 96, 2, height, width],
    ]
    return CACHE_SIZE_0, CACHE_SIZE_N

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


class _VAEWanEncoder0(torch.nn.Module):
    """Wrapper module for Wan VAE encoder extraction."""

    def __init__(self, vae: AutoencoderKLWan, height=704, width=1280):
        super().__init__()
        self.encoder = vae.encoder
        self.cache_dims = get_cache_size_enc(height, width)[0]
        self.clear_cache(vae)

    def forward(self, x, *args) -> torch.Tensor:
        out = self.encoder(x, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        # post-process: update rbln cache tensors
        feat_cache_0 = self._enc_feat_map[0]
        feat_cache_0 = torch.nn.functional.pad(feat_cache_0, (0, 0, 0, 0, 1, 0))  # pad one frame earlier
        dummy_outs = []
        position = torch.tensor(
            0, dtype=torch.int16
        )  # 0 is dummy value -> first output of next chunk have to slice out this frame
        axis = torch.tensor(2, dtype=torch.int16)
        for cache, feat_cache_item, cache_dim in zip(list(args)[1:], self._enc_feat_map[1:], self.cache_dims[1:]):
            n, c, d, h, w = feat_cache_item.shape
            feat_cache_item = feat_cache_item.reshape(n, c, d, -1)
            if cache_dim[2] == 2:
                feat_cache_item = torch.nn.functional.pad(feat_cache_item, (0, 0, 1, 0))  # pad one frame earlier

            dummy_out = torch.ops.rbln_custom_ops.rbln_cache_update(cache, feat_cache_item, position, axis)
            dummy_outs.append(dummy_out)
            print(cache.shape, feat_cache_item.shape)
        return out, feat_cache_0, dummy_outs

    def clear_cache(self, vae):
        self._enc_conv_num = vae._cached_conv_counts["encoder"]
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num # None은 trace가 안되지 않나? -> 되네..
        # self._enc_feat_map = [-1] * self._enc_conv_num

class _VAEWanEncoderN(torch.nn.Module):
    """Wrapper module for Wan VAE encoder extraction."""

    def __init__(self, vae: AutoencoderKLWan, height=704, width=1280):
        super().__init__()
        self.encoder = vae.encoder
        self.cache_dims = get_cache_size_enc(height, width)[1]
        # self.clear_cache(vae) # 이게 필요없네... 사실상 _enc_feat_map이 우리 dram과 동일한 역할

    def forward(self, x, *args) -> torch.Tensor:
        feat_cache_reshaped = []
        
        feat_cache_reshaped.append(args[0])
        # pre-process: reshape rbln cache tensors to torch layout
        for cache, cache_dim in zip(list(args)[1:], self.cache_dims[1:]):
            reshaped_cache = cache.reshape(*cache_dim)  # n c d (hw) -> n c d h w
            feat_cache_reshaped.append(reshaped_cache)
        
        feat_idx = torch.zeros(1, dtype=torch.int32)
        out = self.encoder(x, feat_cache=feat_cache_reshaped, feat_idx=feat_idx)

        # post-process: update rbln cache tensors
        dummy_outs = []
        position = torch.tensor(0, dtype=torch.int16)
        axis = torch.tensor(2, dtype=torch.int16)
        for cache, feat_cache_e1_item in zip(list(args)[1:], feat_cache_reshaped[1:]):
            n, c, d, h, w = feat_cache_e1_item.shape
            feat_cache_e1_item = feat_cache_e1_item.reshape(n, c, d, -1)
            dummy_out = torch.ops.rbln_custom_ops.rbln_cache_update(cache, feat_cache_e1_item, position, axis)
            dummy_outs.append(dummy_out)
            # print(cache.shape, feat_cache_e1_item.shape)
        return out, feat_cache_reshaped[0], dummy_outs

class _VAEWanDecoder0(torch.nn.Module):
    """Wrapper module for Wan VAE decoder extraction."""

    def __init__(self, vae: AutoencoderKLWan, height=704, width=1280):
        super().__init__()
        self.decoder = vae.decoder
        self.cache_dims = get_cache_size_dec(height, width)[0]
        self.clear_cache(vae)


    def forward(self, x, *args) -> torch.Tensor:
        out = self.decoder(x, feat_cache=self._feat_map, feat_idx=self._conv_idx)
        # post-process: update rbln cache tensors
        dummy_outs = []
        position = torch.tensor(
            0, dtype=torch.int16
        )  # 0 is dummy value -> first output of next chunk have to slice out this frame
        axis = torch.tensor(2, dtype=torch.int16)
        for cache, feat_cache_item in zip(list(args), self._feat_map):
            n, c, d, h, w = feat_cache_item.shape
            feat_cache_item = feat_cache_item.reshape(n, c, d, -1)
            feat_cache_item = torch.nn.functional.pad(feat_cache_item, (0, 0, 1, 0))  # pad one frame earlier

            dummy_out = torch.ops.rbln_custom_ops.rbln_cache_update(cache, feat_cache_item, position, axis)
            dummy_outs.append(dummy_out)
            print(cache.shape, feat_cache_item.shape)
        return out, dummy_outs

    def clear_cache(self, vae):
        self._conv_num = vae._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num # None은 trace가 안되지 않나? -> 되네..
        # self._enc_feat_map = [-1] * self._enc_conv_num


class _VAEWanDecoderN(torch.nn.Module):
    """Wrapper module for Wan VAE encoder extraction."""

    def __init__(self, vae: AutoencoderKLWan, height=704, width=1280):
        super().__init__()
        self.decoder = vae.decoder
        self.cache_dims = get_cache_size_dec(height, width)[0]

    def forward(self, x, *args) -> torch.Tensor:
        feat_cache_reshaped = []
        
        # pre-process: reshape rbln cache tensors to torch layout
        for cache, cache_dim in zip(list(args), self.cache_dims):
            reshaped_cache = cache.reshape(*cache_dim)  # n c d (hw) -> n c d h w
            feat_cache_reshaped.append(reshaped_cache)

        feat_idx = torch.zeros(1, dtype=torch.int32)
        out = self.decoder(x, feat_cache=feat_cache_reshaped, feat_idx=feat_idx)

        # post-process: update rbln cache tensors
        dummy_outs = []
        position = torch.tensor(0, dtype=torch.int16)
        axis = torch.tensor(2, dtype=torch.int16)
        for cache, feat_cache_dn_item in zip(list(args), feat_cache_reshaped):
            n, c, d, h, w = feat_cache_dn_item.shape
            feat_cache_dn_item = feat_cache_dn_item.reshape(n, c, d, -1)
            # feat_cache_dn_item = torch.nn.functional.pad(feat_cache_dn_item, (0, 0, 1, 0)) # 이거 안해줘도 되나..? 디코더는..? -> 애초에 cache가 2프레임이니까 패딩 필요 없음

            dummy_out = torch.ops.rbln_custom_ops.rbln_cache_update(cache, feat_cache_dn_item, position, axis)
            dummy_outs.append(dummy_out)
            print(cache.shape, feat_cache_dn_item.shape)
        return out, dummy_outs

    def clear_cache(self, vae):
        self._conv_num = vae._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num # None은 trace가 안되지 않나? -> 되네..


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
            self.encoder_0 = RBLNRuntimeWanVAEEncoder(
                runtime=self.model[0], main_input_name="x", use_slicing=self.rbln_config.use_slicing
            )
            self.encoder_n = RBLNRuntimeWanVAEEncoder(
                runtime=self.model[1], main_input_name="x", use_slicing=self.rbln_config.use_slicing
            )
        self.decoder_0 = RBLNRuntimeWanVAEDecoder(
            runtime=self.model[-2], main_input_name="z", use_slicing=self.rbln_config.use_slicing
        )
        self.decoder_n = RBLNRuntimeWanVAEDecoder(
            runtime=self.model[-1], main_input_name="z", use_slicing=self.rbln_config.use_slicing
        )
        self.image_size = self.rbln_config.image_size
        self.use_slicing = False
        self.use_tiling = False

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNAutoencoderKLWanConfig) -> torch.nn.Module:
        decoder_model_0 = _VAEWanDecoder0(model)
        decoder_model_0.eval()
        
        decoder_model_n = _VAEWanDecoderN(model)
        decoder_model_n.eval()

        if rbln_config.uses_encoder:
            encoder_model_0 = _VAEWanEncoder0(model)
            encoder_model_0.eval()
            
            encoder_model_n = _VAEWanEncoderN(model)
            encoder_model_n.eval()

            return (encoder_model_0, encoder_model_n), (decoder_model_0, decoder_model_n)
        else:
            return (decoder_model_0, decoder_model_n)

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNAutoencoderKLWanConfig) -> Dict[str, rebel.RBLNCompiledModel]:
        compiled_models = {}
        context = CompileContext(use_weight_sharing=False)
        if rbln_config.uses_encoder:
            encoder_models, decoder_models = cls._wrap_model_if_needed(model, rbln_config)
            context, enc0_example_inputs, encn_example_inputs = cls.get_enc_compile_cfg(context, rbln_config)
            enc_compiled_model_0 = cls.compile(
                encoder_models[0],
                rbln_compile_config=rbln_config.compile_cfgs[0],
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device_map["encoder_0"],
                example_inputs=enc0_example_inputs,
                compile_context=context,
            )
            compiled_models["encoder_0"] = enc_compiled_model_0
            enc_compiled_model_n = cls.compile(
                encoder_models[1],
                rbln_compile_config=rbln_config.compile_cfgs[1],
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device_map["encoder_n"],
                example_inputs=encn_example_inputs,
                compile_context=context,
            )
            compiled_models["encoder_n"] = enc_compiled_model_n

        """
        decoder_models = cls._wrap_model_if_needed(model, rbln_config)

        dec_compiled_model_0 = cls.compile(
            decoder_models[0],
            rbln_compile_config=rbln_config.compile_cfgs[-1],
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device_map["decoder"],
        )
        compiled_models["decoder_0"] = dec_compiled_model_0

        dec_compiled_model_n = cls.compile(
            decoder_models[1],
            rbln_compile_config=rbln_config.compile_cfgs[-1],
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device_map["decoder"],
        )
        compiled_models["decoder_n"] = dec_compiled_model_n
        """
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

    @classmethod
    def get_enc_compile_cfg(cls, context, rbln_config):
        encoder_0_compile_config = rbln_config.compile_cfgs[0]
        encoder_n_compile_config = rbln_config.compile_cfgs[1]

        enc0_example_inputs = encoder_0_compile_config.get_dummy_inputs(fill=0)

        # Mark encoder_0's static tensors (cache)
        static_tensors = {}
        for (name, _, _), tensor in zip(encoder_0_compile_config.input_info, enc0_example_inputs):
            if ("feat_cache" in name) and (not "feat_cache_0" in name):
                static_tensors[name] = tensor
                context.mark_static_address(tensor)

        encn_example_inputs = encoder_n_compile_config.get_dummy_inputs(fill=0, static_tensors=static_tensors)
        # Mark encoder_n's static tensors (cache)
        for (name, _, _), tensor in zip(encoder_n_compile_config.input_info, encn_example_inputs):
            if ("feat_cache" in name) and (not "feat_cache_0" in name):
                context.mark_static_address(tensor)
        return context, enc0_example_inputs, encn_example_inputs

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
            cache_0, cache_1 = get_cache_size_enc(rbln_config.height, rbln_config.width)
            for i, (shape_0, shape_1) in enumerate(zip(cache_0, cache_1)):
                if i > 0:
                    shape_0 = [*shape_0[:3], shape_0[-2] * shape_0[-1]] # N C D HW # FIXME H,W 도 support 가능?
                    shape_1 = [*shape_1[:3], shape_1[-2] * shape_1[-1]]
                vae_enc_0_input_info.append((f"feat_cache_{i}", shape_0, "float32"))
                vae_enc_1_input_info.append((f"feat_cache_{i}", shape_1, "float32"))

            compile_cfgs.append(RBLNCompileConfig(compiled_model_name="encoder_0", input_info=vae_enc_0_input_info))
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name="encoder_n", input_info=vae_enc_1_input_info))

        rbln_config.vae_scale_factor_temporal = rbln_config.vae_scale_factor_temporal or 4 # tmp code
        rbln_config.vae_scale_factor_spatial = rbln_config.vae_scale_factor_spatial or 8 # tmp code
        
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
            # expected_models = ["encoder_0", "encoder_n", "decoder_0", "decoder_n"]
            expected_models = ["encoder_0", "encoder_n"] # tmp code

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
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
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
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)

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
        # FIXME(seinpark) super()._encode(x)?
        _, _, num_frame, height, width = x.shape

        self.clear_cache()
        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)

        # if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
        #     return self.tiled_encode(x)

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
