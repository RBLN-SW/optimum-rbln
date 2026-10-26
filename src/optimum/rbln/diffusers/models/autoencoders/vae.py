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

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import QwenImage21ResidualDownBlock
from diffusers.models.autoencoders.autoencoder_kl_wan import patchify as wan_patchify
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify as wan_unpatchify
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution, IdentityDistribution

from ....utils.runtime_utils import RBLNPytorchRuntime


if TYPE_CHECKING:
    from diffusers import (
        AutoencoderKL,
        AutoencoderKLCosmos,
        AutoencoderKLQwenImage21,
        AutoencoderKLTemporalDecoder,
        VQModel,
    )


class RBLNRuntimeVAEEncoder(RBLNPytorchRuntime):
    def encode(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        moments = self.forward(x.contiguous())
        posterior = DiagonalGaussianDistribution(moments)
        return posterior


class RBLNRuntimeVAEDecoder(RBLNPytorchRuntime):
    def decode(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.forward(z)


class RBLNRuntimeCosmosVAEEncoder(RBLNPytorchRuntime):
    def encode(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.forward(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.forward(x)
        posterior = IdentityDistribution(h)
        return posterior


class RBLNRuntimeCosmosVAEDecoder(RBLNPytorchRuntime):
    def decode(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self.forward(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self.forward(z)
        return decoded


class RBLNRuntimeWanVAEEncoder(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name", "encoder_n", "patch_size", "dtype"]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        if self.patch_size is not None:
            x = wan_patchify(x, patch_size=self.patch_size)

        _, _, num_frame, _, _ = x.shape
        outs = []
        feat_cache_0 = None
        for i in range(1 + (num_frame - 1) // 4):
            if i == 0:
                ret = self.forward(x[:, :, :1, :, :])
            else:
                ret = self.encoder_n(x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :], feat_cache_0)
            out_i, feat_cache_0 = ret[0], ret[1]
            outs.append(out_i)

        return torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]


class RBLNRuntimeWanVAEDecoder(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name", "decoder_n", "patch_size", "dtype", "post_quant_conv"]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(self.dtype)
        if self.post_quant_conv is not None:
            z = self.post_quant_conv.to(z.dtype)(z)

        _, _, num_frame, _, _ = z.shape
        outs = []
        feat_cache_0 = None
        for i in range(num_frame):
            if i == 0:
                ret = self.forward(z[:, :, :1, :, :])
            else:
                ret = self.decoder_n(z[:, :, i : i + 1, :, :], feat_cache_0)
            out_i, feat_cache_0 = ret[0], ret[1]
            outs.append(out_i)

        out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
        if self.patch_size is not None:
            out = wan_unpatchify(out, patch_size=self.patch_size)
        return torch.clamp(out, min=-1.0, max=1.0)


class _VAEDecoder(torch.nn.Module):
    def __init__(self, vae: "AutoencoderKL"):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        vae_out = self.vae.decode(z, return_dict=False)
        return vae_out


class _VAETemporalDecoder(torch.nn.Module):
    def __init__(self, vae: "AutoencoderKLTemporalDecoder"):
        super().__init__()
        self.vae = vae
        self.num_frames = None

    def forward(self, z):
        vae_out = self.vae.decode(z, num_frames=self.num_frames, return_dict=False)
        return vae_out


class _VAEEncoder(torch.nn.Module):
    def __init__(self, vae: Union["AutoencoderKL", "AutoencoderKLTemporalDecoder"]):
        super().__init__()
        self.vae = vae

    def encode(self, x: torch.FloatTensor, return_dict: bool = True):
        if hasattr(self, "use_tiling") and hasattr(self, "use_slicing"):
            if self.use_tiling and (
                x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size
            ):
                return self.tiled_encode(x, return_dict=return_dict)

            if self.use_slicing and x.shape[0] > 1:
                encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
                h = torch.cat(encoded_slices)
            else:
                h = self.encoder(x)
                if self.quant_conv is not None:
                    h = self.quant_conv(h)

        else:
            h = self.encoder(x)
            if self.quant_conv is not None:
                h = self.quant_conv(h)
        return h

    def forward(self, x):
        vae_out = _VAEEncoder.encode(self.vae, x, return_dict=False)
        return vae_out


class _VAECosmosEncoder(torch.nn.Module):
    def __init__(self, vae: "AutoencoderKLCosmos"):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        vae_out = self.vae._encode(x)
        return vae_out


class _VAECosmosDecoder(torch.nn.Module):
    def __init__(self, vae: "AutoencoderKLCosmos"):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        vae_out = self.vae._decode(z, return_dict=False)
        return vae_out


# Additive attention bias for keys outside the image; it vanishes in the softmax in bf16 and fp32.
_QWENIMAGE21_KEY_MASK_BIAS = -1e9


def _get_qwenimage21_mask(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # The image sits at the top left of the canvas and its size is a multiple of the VAE scale factor, so the mask at
    # any resolution of the encoder is a strided slice of the full one.
    stride = mask.shape[-2] // x.shape[-2]
    return mask[..., ::stride, ::stride].to(x.dtype)


def _apply_qwenimage21_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x * _get_qwenimage21_mask(mask, x)


def _run_qwenimage21_residual_block(block, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # QwenImage21ResidualBlock.forward for one frame, masked after every convolution.
    h = _apply_qwenimage21_mask(block.conv_shortcut(x), mask)
    x = _apply_qwenimage21_mask(block.conv1(block.nonlinearity(block.norm1(x))), mask)
    x = _apply_qwenimage21_mask(block.conv2(block.dropout(block.nonlinearity(block.norm2(x)))), mask)
    return x + h


def _run_qwenimage21_attention_block(block, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # QwenImage21AttentionBlock.forward with the canvas outside the image left out as keys.
    identity = x
    batch_size, channels, time, height, width = x.size()
    x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
    x = block.norm(x)
    qkv = block.to_qkv(x)
    qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
    qkv = qkv.permute(0, 1, 3, 2).contiguous()
    q, k, v = qkv.chunk(3, dim=-1)
    key_mask = _get_qwenimage21_mask(mask, identity).reshape(1, 1, 1, height * width)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=(1.0 - key_mask) * _QWENIMAGE21_KEY_MASK_BIAS)
    x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
    x = block.proj(x)
    x = x.view(batch_size, time, channels, height, width)
    x = x.permute(0, 2, 1, 3, 4)
    return _apply_qwenimage21_mask(x + identity, mask)


class _VAEQwenImage21Encoder(torch.nn.Module):
    """`vae._encode` of one image at the top left of a larger canvas, with the result of the image alone.

    The canvas lets one compiled graph encode every image size that fits it. `mask` is 1 over the image and 0
    elsewhere. Every convolution of the image alone reads zero padding past the image border, at each layer; so each
    module output that can turn the canvas outside the image non-zero is multiplied by the mask at its resolution, and
    the mid-block attention leaves the outside out as keys. The upstream modules run in the order of `_encode` for one
    frame, whose feature cache leaves the result unchanged.

    The mask is an argument, never module state: compilation clones the module first and would detach any state read
    through hooks.
    """

    def __init__(self, vae: "AutoencoderKLQwenImage21"):
        super().__init__()
        if vae.config.patch_size is not None or not all(
            isinstance(block, QwenImage21ResidualDownBlock) for block in vae.encoder.down_blocks
        ):
            raise NotImplementedError("The RBLN VAE encoder covers the residual encoder without patchify.")
        self.vae = vae

    def forward(self, x, mask):
        encoder = self.vae.encoder
        x = _apply_qwenimage21_mask(encoder.conv_in(x * mask), mask)
        for block in encoder.down_blocks:
            # QwenImage21ResidualDownBlock.forward; the downsampler's time_conv runs only with a cached frame.
            shortcut = _apply_qwenimage21_mask(block.avg_shortcut(x), mask)
            for resnet in block.resnets:
                x = _run_qwenimage21_residual_block(resnet, x, mask)
            if block.downsampler is not None:
                x = _apply_qwenimage21_mask(block.downsampler(x), mask)
            x = x + shortcut
        mid = encoder.mid_block
        x = _run_qwenimage21_residual_block(mid.resnets[0], x, mask)
        for attention, resnet in zip(mid.attentions, mid.resnets[1:], strict=True):
            if attention is not None:
                x = _run_qwenimage21_attention_block(attention, x, mask)
            x = _run_qwenimage21_residual_block(resnet, x, mask)
        x = _apply_qwenimage21_mask(encoder.conv_out(encoder.nonlinearity(encoder.norm_out(x))), mask)
        return _apply_qwenimage21_mask(self.vae.quant_conv(x), mask)


class _VAEQwenImage21Decoder(torch.nn.Module):
    # One latent frame: the decoder's frame loop runs once and its feature cache stays empty.
    def __init__(self, vae: "AutoencoderKLQwenImage21"):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae._decode(z, return_dict=False)[0]


class RBLNRuntimeVQEncoder(RBLNPytorchRuntime):
    def encode(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        h = self.forward(x.contiguous())
        return h


class RBLNRuntimeVQDecoder(RBLNPytorchRuntime):
    def decode(self, h: torch.Tensor, force_not_quantize: bool = False, shape=None, **kwargs) -> list[torch.Tensor]:
        if not (force_not_quantize and not self.lookup_from_codebook):
            raise ValueError(
                "Currently, the `decode` method of the class `RBLNVQModel` is executed successfully only if `force_not_quantize` is True and `config.lookup_from_codebook` is False"
            )
        commit_loss = torch.zeros(h.shape[0]).to(h.device, dtype=h.dtype)
        dec = self.forward(h.contiguous())
        return dec, commit_loss


class _VQEncoder(torch.nn.Module):
    def __init__(self, vq_model: "VQModel"):
        super().__init__()
        self.vq_model = vq_model

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        h = self.vq_model.encoder(x)
        h = self.vq_model.quant_conv(h)
        return h

    def forward(self, x: torch.Tensor):
        vq_out = self.encode(x)
        return vq_out


class _VQDecoder(torch.nn.Module):
    def __init__(self, vq_model: "VQModel"):
        super().__init__()
        self.vq_model = vq_model

    def decode(self, h: torch.Tensor, force_not_quantize: bool = False, return_dict: bool = True, shape=None):
        quant = h
        quant2 = self.vq_model.post_quant_conv(quant)
        quant = quant if self.vq_model.config.norm_type == "spatial" else None
        dec = self.vq_model.decoder(quant2, quant)
        return dec

    def forward(self, h: torch.Tensor):
        vq_out = self.decode(h)
        return vq_out
