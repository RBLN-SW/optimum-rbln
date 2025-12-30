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

from typing import TYPE_CHECKING, List, Union

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution, IdentityDistribution

from ....utils.runtime_utils import RBLNPytorchRuntime


if TYPE_CHECKING:
    from diffusers import AutoencoderKL, AutoencoderKLCosmos, AutoencoderKLTemporalDecoder, VQModel


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


class RBLNRuntimeVQEncoder(RBLNPytorchRuntime):
    def encode(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        h = self.forward(x.contiguous())
        return h


class RBLNRuntimeVQDecoder(RBLNPytorchRuntime):
    def decode(self, h: torch.Tensor, force_not_quantize: bool = False, shape=None, **kwargs) -> List[torch.Tensor]:
        if not (force_not_quantize and not self.lookup_from_codebook):
            raise ValueError(
                "Currently, the `decode` method of the class `RBLNVQModel` is executed successfully only if `force_not_quantize` is True and `config.lookup_from_codebook` is False"
            )
        commit_loss = torch.zeros((h.shape[0])).to(h.device, dtype=h.dtype)
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
