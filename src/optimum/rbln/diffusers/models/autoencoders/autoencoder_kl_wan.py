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

from typing import TYPE_CHECKING, Union

import rebel
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan, patchify, unpatchify
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from rebel.compile_context import CompileContext

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
        [s[0], s[1], s[2] if i in NO_PAD_INDICES else PADDED_FRAME, s[3], s[4]] for i, s in enumerate(CACHE_SIZE_0)
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


def _to_cdhw_cache(x: torch.Tensor) -> torch.Tensor:
    # channel-first (n,c,d,h,w) -> cache (n, d, c*h*w). Folding C into the merged c*h*w axis means there
    # is no standalone channel dim for the device to 64-block-pad (NCDHW64c): the stored cache carries the
    # true c*h*w (a multiple of 64), so the read and write physical views match exactly. This avoids both
    # the AnnotatePhysicalView reconcile (channel-last stored C=128 padded vs read C=96) and the SHM
    # blow-up (channel-last needs a transpose+aligned_pad chain; c*h*w needs neither). W never scrambles.
    n, c, d, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(n, d, c * h * w).contiguous()


def _from_cdhw_cache(cache: torch.Tensor, c: int, h: int, w: int) -> torch.Tensor:
    # cache (n, d, c*h*w) -> channel-first (n, c, d, h, w).
    n, d, _chw = cache.shape
    return cache.reshape(n, d, c, h, w).permute(0, 2, 1, 3, 4)


class _VAEWanEncoder0(torch.nn.Module):
    """Wrapper module for Wan VAE encoder extraction."""

    def __init__(self, vae: AutoencoderKLWan, height=704, width=1280):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv  # 1x1x1 pointwise -> fold into the graph (per-chunk == concat-then-quant)
        self.cache_dims = get_cache_size_enc(height, width)[0]
        self.clear_cache(vae)

    def forward(self, x, *args) -> torch.Tensor:
        # E0 (first chunk): the encoder generates all conv feat-caches. idx 1.. are written to shared
        # device-resident static DRAM via rbln_cache_update; idx 0 concatenates with the f32 input at
        # conv_in downstream, so it is a runtime output. Caches are stored CHANNEL-LAST (n,d,h,w,c):
        # only C gets 64-aligned padding, so W is never scrambled (flat/NCHW gave pearson 0.98).
        out = self.encoder(x, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        out = self.quant_conv(out)
        position = torch.tensor(0, dtype=torch.int16)  # 0 is dummy; next chunk slices this frame out
        axis = torch.tensor(1, dtype=torch.int16)  # cdhw (n,d,c*h*w): frame D is axis 1
        dummy_outs = []
        for cache, feat_cache_item, cache_dim in zip(
            list(args)[1:], self._enc_feat_map[1 : len(self.cache_dims)], self.cache_dims[1:], strict=False
        ):
            if cache_dim[2] == 2:
                feat_cache_item = torch.nn.functional.pad(feat_cache_item, (0, 0, 0, 0, 1, 0))  # pad D 1->2
            feat_cache_item = _to_cdhw_cache(feat_cache_item)  # (n,c,d,h,w) -> (n,d,c*h*w)
            dummy_outs.append(torch.ops.rbln_custom_ops.rbln_cache_update(cache, feat_cache_item, position, axis))
        # idx 0: runtime output, kept CHANNEL-FIRST (n,c,d,h,w) -- it is I/O, not static, so no permute.
        fc0 = torch.nn.functional.pad(self._enc_feat_map[0], (0, 0, 0, 0, 1, 0)).contiguous()  # pad D 1->2
        return out, fc0, dummy_outs

    def clear_cache(self, vae):
        self._enc_conv_num = vae._cached_conv_counts["encoder"]
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


class _VAEWanEncoderN(torch.nn.Module):
    """Wrapper module for Wan VAE encoder extraction."""

    def __init__(self, vae: AutoencoderKLWan, height=704, width=1280):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv  # 1x1x1 pointwise -> fold into the graph
        self.cache_dims = get_cache_size_enc(height, width)[1]

    def forward(self, x, *args) -> torch.Tensor:
        # EN (steady state): idx 0 is a runtime I/O input, already CHANNEL-FIRST (n,c,d,h,w) -- used
        # directly, no permute. idx 1.. are read from shared static DRAM (channel-last -> channel-first).
        # After the encoder, write idx 1.. back channel-last via rbln_cache_update; idx 0 is returned
        # channel-first for the next chunk (no layout flip).
        feat_cache_reshaped = [args[0]]  # idx0 already channel-first (n,c,d,h,w)
        for i, cache in enumerate(list(args)[1:], start=1):
            c_, h_, w_ = self.cache_dims[i][1], self.cache_dims[i][3], self.cache_dims[i][4]
            feat_cache_reshaped.append(_from_cdhw_cache(cache, c_, h_, w_))  # (n,d,c*h*w) -> (n,c,d,h,w)

        feat_idx = torch.zeros(1, dtype=torch.int32)
        out = self.encoder(x, feat_cache=feat_cache_reshaped, feat_idx=feat_idx)
        out = self.quant_conv(out)

        position = torch.tensor(0, dtype=torch.int16)
        axis = torch.tensor(1, dtype=torch.int16)  # cdhw (n,d,c*h*w): frame D is axis 1
        dummy_outs = []
        for cache, item in zip(list(args)[1:], feat_cache_reshaped[1:], strict=False):
            item = _to_cdhw_cache(item)  # (n,c,d,h,w) -> (n,d,c*h*w)
            dummy_outs.append(torch.ops.rbln_custom_ops.rbln_cache_update(cache, item, position, axis))
        return out, feat_cache_reshaped[0].contiguous(), dummy_outs  # idx0 channel-first (no flip)


class _VAEWanDecoder0(torch.nn.Module):
    """Wrapper module for Wan VAE decoder extraction."""

    def __init__(self, vae: AutoencoderKLWan, height=704, width=1280):
        super().__init__()
        self.decoder = vae.decoder
        self.cache_dims = get_cache_size_dec(height, width)[0]
        self.clear_cache(vae)

    def forward(self, x, *args) -> torch.Tensor:
        # D0 (first chunk): the decoder generates all conv feat-caches. idx 1.. are written to shared
        # device-resident static DRAM via rbln_cache_update (CHANNEL-LAST (n,d,h,w,c): only C gets
        # 64-aligned padding, W is never scrambled); idx 0 concatenates with the f32 latent at conv_in
        # so it is a runtime output. post_quant_conv is applied on the host before the loop (folding it
        # into DN makes post_quant_conv + upsample3d + idx0-output fail the RblnTensor layout pass).
        out = self.decoder(x, feat_cache=self._feat_map, feat_idx=self._conv_idx, first_chunk=True)
        position = torch.tensor(0, dtype=torch.int16)  # 0 is dummy; next chunk slices this frame out
        axis = torch.tensor(1, dtype=torch.int16)  # cdhw (n,d,c*h*w): frame D is axis 1
        dummy_outs = []
        for cache, feat_cache_item, _cache_dim in zip(
            list(args), self._feat_map[1 : len(self.cache_dims)], self.cache_dims[1:], strict=False
        ):
            if isinstance(feat_cache_item, str):
                # diffusers upsample3d stores a "Rep" string on the first chunk (no real cache). Write
                # zeros (from the static input, no big SHM const) so DN's time_conv(x, 0) == the "Rep"
                # no-context path exactly.
                feat_cache_item = cache * 0.0  # cdhw (n,d,c*h*w) zeros (same shape as the cache slot)
            else:
                feat_cache_item = torch.nn.functional.pad(feat_cache_item, (0, 0, 0, 0, 1, 0))  # pad D 1->2
                feat_cache_item = _to_cdhw_cache(feat_cache_item)  # (n,c,d,h,w) -> (n,d,c*h*w)
            dummy_outs.append(torch.ops.rbln_custom_ops.rbln_cache_update(cache, feat_cache_item, position, axis))
        # idx 0: runtime output, kept CHANNEL-FIRST (n,c,d,h,w) -- I/O, not static, so no permute.
        fc0 = torch.nn.functional.pad(self._feat_map[0], (0, 0, 0, 0, 1, 0)).contiguous()  # pad D 1->2
        return out, fc0, dummy_outs

    def clear_cache(self, vae):
        self._conv_num = vae._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num


class _VAEWanDecoderN(torch.nn.Module):
    """Wrapper module for Wan VAE decoder extraction."""

    def __init__(self, vae: AutoencoderKLWan, height=704, width=1280):
        super().__init__()
        self.decoder = vae.decoder
        self.cache_dims = get_cache_size_dec(height, width)[1]

    def forward(self, x, *args) -> torch.Tensor:
        # DN (steady state): idx 0 is a runtime input; idx 1.. are read from shared static DRAM. Read
        # each channel-last (n,d,h,w,c) back to (n,c,d,h,w), run the decoder, then write the updated
        # caches back channel-last via rbln_cache_update. idx 0 is a runtime output. post_quant_conv is
        # applied on the host before the loop (see _VAEWanDecoder0).
        feat_cache_reshaped = [args[0]]  # idx0 already channel-first (n,c,d,h,w), runtime I/O
        for i, cache in enumerate(list(args)[1:], start=1):
            c_, h_, w_ = self.cache_dims[i][1], self.cache_dims[i][3], self.cache_dims[i][4]
            feat_cache_reshaped.append(_from_cdhw_cache(cache, c_, h_, w_))  # (n,d,c*h*w) -> (n,c,d,h,w)

        feat_idx = torch.zeros(1, dtype=torch.int32)
        out = self.decoder(x, feat_cache=feat_cache_reshaped, feat_idx=feat_idx)

        position = torch.tensor(0, dtype=torch.int16)
        axis = torch.tensor(1, dtype=torch.int16)  # cdhw (n,d,c*h*w): frame D is axis 1
        dummy_outs = []
        for cache, item in zip(list(args)[1:], feat_cache_reshaped[1:], strict=False):
            item = _to_cdhw_cache(item)  # (n,c,d,h,w) -> (n,d,c*h*w)
            dummy_outs.append(torch.ops.rbln_custom_ops.rbln_cache_update(cache, item, position, axis))
        return out, feat_cache_reshaped[0].contiguous(), dummy_outs  # idx0 channel-first (no flip)


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

        # post_quant_conv weights (persisted in config) -> rebuild for host-side application in _decode.
        pqc_w = getattr(self.rbln_config, "post_quant_conv_weight", None)
        if pqc_w is not None:
            pqc_b = getattr(self.rbln_config, "post_quant_conv_bias", None)
            self._pqc_weight = torch.tensor(pqc_w, dtype=torch.float32)
            self._pqc_bias = torch.tensor(pqc_b, dtype=torch.float32) if pqc_b is not None else None
        else:
            self._pqc_weight = None
            self._pqc_bias = None

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNAutoencoderKLWanConfig) -> torch.nn.Module:
        h, w = rbln_config.height, rbln_config.width
        decoder_model_0 = _VAEWanDecoder0(model, height=h, width=w)
        decoder_model_0.eval()

        decoder_model_n = _VAEWanDecoderN(model, height=h, width=w)
        decoder_model_n.eval()

        if rbln_config.uses_encoder:
            encoder_model_0 = _VAEWanEncoder0(model, height=h, width=w)
            encoder_model_0.eval()

            encoder_model_n = _VAEWanEncoderN(model, height=h, width=w)
            encoder_model_n.eval()

            return (encoder_model_0, encoder_model_n), (decoder_model_0, decoder_model_n)
        else:
            return (decoder_model_0, decoder_model_n)

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNAutoencoderKLWanConfig) -> dict[str, rebel.RBLNCompiledModel]:
        compiled_models = {}
        # E0 and EN wrap the SAME vae.encoder -> share its weights on device (use_weight_sharing).
        context = CompileContext(use_weight_sharing=True)
        if rbln_config.uses_encoder:
            encoder_models, decoder_models = cls._wrap_model_if_needed(model, rbln_config)
            context, enc0_example_inputs, encn_example_inputs = cls.get_enc_compile_cfg(context, rbln_config)
            enc_compiled_model_0 = rebel.compile_from_torch(
                encoder_models[0],
                input_info=rbln_config.compile_cfgs[0].input_info,
                example_inputs=enc0_example_inputs,
                compile_context=context,
            )
            compiled_models["encoder_0"] = enc_compiled_model_0
            enc_compiled_model_n = rebel.compile_from_torch(
                encoder_models[1],
                input_info=rbln_config.compile_cfgs[1].input_info,
                example_inputs=encn_example_inputs,
                compile_context=context,
            )
            compiled_models["encoder_n"] = enc_compiled_model_n
            dec_models = decoder_models
        else:
            dec_models = cls._wrap_model_if_needed(model, rbln_config)

        # decoder gets its OWN CompileContext (separate from the encoder's static-cache context).
        # D0 and DN wrap the SAME vae.decoder -> share its weights on device (use_weight_sharing).
        context = CompileContext(use_weight_sharing=True)
        context, dec0_example_inputs, decn_example_inputs = cls.get_dec_compile_cfg(context, rbln_config)
        dec_compiled_model_0 = cls.compile(
            dec_models[0],
            rbln_compile_config=rbln_config.compile_cfgs[-2],
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device_map["decoder_0"],
            example_inputs=dec0_example_inputs,
            compile_context=context,
        )
        compiled_models["decoder_0"] = dec_compiled_model_0

        dec_compiled_model_n = cls.compile(
            dec_models[1],
            rbln_compile_config=rbln_config.compile_cfgs[-1],
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device_map["decoder_n"],
            example_inputs=decn_example_inputs,
            compile_context=context,
        )
        compiled_models["decoder_n"] = dec_compiled_model_n
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
        # feat_cache idx 1.. are shared device-resident static DRAM across E0 and EN (idx 0 is runtime
        # I/O). The SAME tensor objects are reused for both graphs (via static_tensors) and marked with
        # mark_static_address so E0's rbln_cache_update writes are visible to EN's reads.
        encoder_0_compile_config = rbln_config.compile_cfgs[0]
        encoder_n_compile_config = rbln_config.compile_cfgs[1]

        enc0_example_inputs = encoder_0_compile_config.get_dummy_inputs(fill=0)
        static_tensors = {}
        for (name, _, _), tensor in zip(encoder_0_compile_config.input_info, enc0_example_inputs, strict=False):
            if ("feat_cache" in name) and ("feat_cache_0" not in name):
                static_tensors[name] = tensor
                context.mark_static_address(tensor)

        encn_example_inputs = encoder_n_compile_config.get_dummy_inputs(fill=0, static_tensors=static_tensors)
        for (name, _, _), tensor in zip(encoder_n_compile_config.input_info, encn_example_inputs, strict=False):
            if ("feat_cache" in name) and ("feat_cache_0" not in name):
                context.mark_static_address(tensor)
        return context, enc0_example_inputs, encn_example_inputs

    @classmethod
    def get_dec_compile_cfg(cls, context, rbln_config):
        # feat_cache idx 1.. are shared device-resident static DRAM across D0 and DN (idx 0 is runtime
        # I/O). D0's input_info has feat_cache_1.. only (idx 0 dropped) -> all of them are static; the
        # SAME tensor objects are reused for DN (via static_tensors) and marked so D0's cache_update
        # writes are visible to DN's reads.
        decoder_0_compile_config = rbln_config.compile_cfgs[-2]
        decoder_n_compile_config = rbln_config.compile_cfgs[-1]

        dec0_example_inputs = decoder_0_compile_config.get_dummy_inputs(fill=0)
        static_tensors = {}
        for (name, _, _), tensor in zip(decoder_0_compile_config.input_info, dec0_example_inputs, strict=False):
            if "feat_cache" in name:
                static_tensors[name] = tensor
                context.mark_static_address(tensor)

        decn_example_inputs = decoder_n_compile_config.get_dummy_inputs(fill=0, static_tensors=static_tensors)
        for (name, _, _), tensor in zip(decoder_n_compile_config.input_info, decn_example_inputs, strict=False):
            if ("feat_cache" in name) and ("feat_cache_0" not in name):
                context.mark_static_address(tensor)
        return context, dec0_example_inputs, decn_example_inputs

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
            # Caches are shared device-resident static DRAM across E0/EN (idx 1..), written by
            # rbln_cache_update and read by the encoder. Stored CHANNEL-LAST (n,c,d,h,w) -> (n,d,h,w,c)
            # so only C is 64-aligned padded (W stays a middle axis; flat/NCHW scrambled W -> pearson 0.98).
            # idx 0 is runtime I/O. E0 and EN share the same cache buffers, so both declare all of them.
            cache_0, cache_1 = get_cache_size_enc(rbln_config.height, rbln_config.width)
            for i, (shape_0, shape_1) in enumerate(zip(cache_0, cache_1, strict=False)):
                n0, c0, d0, h0, w0 = shape_0
                n1, c1, d1, h1, w1 = shape_1
                if i == 0:
                    # idx 0 is runtime I/O (handed off between chunks), NOT static DRAM -> keep it
                    # channel-first (n,c,d,h,w); the W-scramble only affects the cache_update/static path.
                    vae_enc_0_input_info.append((f"feat_cache_{i}", [n0, c0, d0, h0, w0], "float32"))
                    vae_enc_1_input_info.append((f"feat_cache_{i}", [n1, c1, d1, h1, w1], "float32"))
                else:
                    vae_enc_0_input_info.append((f"feat_cache_{i}", [n0, d0, c0 * h0 * w0], "float32"))  # cdhw (n,d,c*h*w)
                    vae_enc_1_input_info.append((f"feat_cache_{i}", [n1, d1, c1 * h1 * w1], "float32"))  # cdhw (n,d,c*h*w)

            compile_cfgs.append(RBLNCompileConfig(compiled_model_name="encoder_0", input_info=vae_enc_0_input_info))
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name="encoder_n", input_info=vae_enc_1_input_info))

        rbln_config.vae_scale_factor_temporal = rbln_config.vae_scale_factor_temporal or 4  # tmp code
        rbln_config.vae_scale_factor_spatial = rbln_config.vae_scale_factor_spatial or 8  # tmp code

        latent_height = rbln_config.height // rbln_config.vae_scale_factor_spatial
        latent_width = rbln_config.width // rbln_config.vae_scale_factor_spatial

        # decoder is chunked: D0 (first latent frame) / DN (each subsequent latent frame). Caches are
        # shared static DRAM across D0/DN (idx 1.., written by rbln_cache_update, read by the decoder),
        # stored CHANNEL-LAST (n,c,d,h,w) -> (n,d,h,w,c) so only C is 64-aligned padded (W never
        # scrambled). idx 0 is runtime I/O (concats with the f32 latent at conv_in). D0 does NOT take a
        # feat_cache_0 input (it generates idx 0 as an output); DN takes idx 0 as a runtime input.
        dec_cache_0, dec_cache_n = get_cache_size_dec(rbln_config.height, rbln_config.width)
        # decoder input channels = VAE z_dim (16), NOT num_channels_latents (the transformer's latent
        # channels, e.g. 24). The VAE decoder conv_in expects z_dim channels.
        z_dim = getattr(model_config, "z_dim", rbln_config.num_channels_latents)
        vae_dec_0_input_info = [
            ("z", [batch_size, z_dim, 1, latent_height, latent_width], "float32"),
        ]
        vae_dec_n_input_info = [
            ("z", [batch_size, z_dim, 1, latent_height, latent_width], "float32"),
        ]
        for i, (shape_0, shape_n) in enumerate(zip(dec_cache_0, dec_cache_n, strict=False)):
            n0, c0, d0, h0, w0 = shape_0
            nn, cn, dn, hn, wn = shape_n
            if i > 0:  # D0 generates idx 0 (runtime output); only idx 1.. are D0 inputs (static, cdhw)
                vae_dec_0_input_info.append((f"feat_cache_{i}", [n0, d0, c0 * h0 * w0], "float32"))  # cdhw (n,d,c*h*w)
                vae_dec_n_input_info.append((f"feat_cache_{i}", [nn, dn, cn * hn * wn], "float32"))  # cdhw (n,d,c*h*w)
            else:  # idx 0 is runtime I/O (DN input) -> channel-first (n,c,d,h,w), no scramble concern
                vae_dec_n_input_info.append((f"feat_cache_{i}", [nn, cn, dn, hn, wn], "float32"))

        compile_cfgs.append(RBLNCompileConfig(compiled_model_name="decoder_0", input_info=vae_dec_0_input_info))
        compile_cfgs.append(RBLNCompileConfig(compiled_model_name="decoder_n", input_info=vae_dec_n_input_info))

        # post_quant_conv (1x1x1 pointwise) is applied on the host before the decode loop rather than
        # folded into the decoder graphs: folding it into DN makes the (post_quant_conv + upsample3d +
        # idx0-output) combination fail the compiler's RblnTensor layout pass. It is tiny (z_dim x z_dim),
        # so its weights are persisted in the config and rebuilt at runtime (survives save/load).
        pqc = model.post_quant_conv
        rbln_config.post_quant_conv_weight = pqc.weight.detach().float().cpu().numpy().tolist()
        rbln_config.post_quant_conv_bias = (
            pqc.bias.detach().float().cpu().numpy().tolist() if pqc.bias is not None else None
        )

        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: list[rebel.RBLNCompiledModel],
        rbln_config: RBLNAutoencoderKLWanConfig,
    ) -> list[rebel.Runtime]:
        if rbln_config.uses_encoder:
            expected_models = ["encoder_0", "encoder_n", "decoder_0", "decoder_n"]
        else:
            expected_models = ["decoder_0", "decoder_n"]

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
        self, x: torch.Tensor, return_dict: bool = True
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
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

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> torch.Tensor | DecoderOutput:
        """
        Decode a latent representation into a video.

        Args:
            z: The latent representation to decode.
            return_dict:
                Whether to return output as a dictionary. Defaults to True.

        Returns:
            The decoded video or DecoderOutput if return_dict=True
        """
        decoded = self._decode(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def _decode(self, z: torch.Tensor):
        # RBLN chunked decode. One latent frame per chunk: the first (D0, first_chunk) yields 1 pixel
        # frame; each subsequent (DN) yields 4 (temporal x4 via upsample3d). All conv feat-caches are
        # handed off as runtime tensors between chunks (channel-first). post_quant_conv (1x1x1 pointwise)
        # is applied here on the host, before the loop -- it cannot be folded into DN (see
        # _VAEWanDecoder0), and being pointwise it commutes with the causal cache concat downstream.
        if self._pqc_weight is not None:
            bias = self._pqc_bias.to(z.dtype) if self._pqc_bias is not None else None
            z = torch.nn.functional.conv3d(z, self._pqc_weight.to(z.dtype), bias)
        _, _, num_frame, _, _ = z.shape
        outs = []
        feat_cache_0 = None
        for i in range(num_frame):
            if i == 0:
                ret = self.decoder_0(z[:, :, :1, :, :])
            else:
                ret = self.decoder_n(z[:, :, i : i + 1, :, :], feat_cache_0)
            out_i, feat_cache_0 = ret[0], ret[1]  # (decoder_out, feat_cache_0, *dummy_cache_updates)
            outs.append(out_i)

        out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
        if self.config.patch_size is not None:
            out = unpatchify(out, patch_size=self.config.patch_size)
        out = torch.clamp(out, min=-1.0, max=1.0)
        return out

    def _encode(self, x: torch.Tensor):
        # RBLN chunked encode. The Wan encoder is causal-temporal: the first latent frame comes from
        # frame 0 (chunk E0), then each subsequent 4 input frames -> 1 latent frame (chunk EN). The idx-0
        # conv cache (feat_cache_0) is handed off as a runtime tensor between chunks; idx 1.. persist on
        # device via shared static DRAM (rbln_cache_update). quant_conv is folded into each chunk's graph.
        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)

        _, _, num_frame, _, _ = x.shape
        iter_ = 1 + (num_frame - 1) // 4
        outs = []
        feat_cache_0 = None
        for i in range(iter_):
            if i == 0:
                ret = self.encoder_0(x[:, :, :1, :, :])
            else:
                ret = self.encoder_n(x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :], feat_cache_0)
            out_i, feat_cache_0 = ret[0], ret[1]  # (encoder_out, feat_cache_0, *dummy_cache_updates)
            outs.append(out_i)

        enc = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
        return enc
