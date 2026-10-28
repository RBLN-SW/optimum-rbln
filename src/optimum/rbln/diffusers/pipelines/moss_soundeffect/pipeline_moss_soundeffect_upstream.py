# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The MOSS-SoundEffect v2 model code and pipeline of https://github.com/OpenMOSS/MOSS-TTS (Apache-2.0),
# `moss_soundeffect_v2` at revision 934d682: `WanAudioModel` and its blocks (`diffsynth/models/wan_audio_dit.py`,
# `wan_video_dit.py`), `DAC` (`diffsynth/models/dac_vae.py`), `FlowMatchScheduler` (`diffsynth/schedulers/flow_match.py`),
# the prompt cleaning of `WanPrompter` and the text-to-audio path of `MossSoundEffectPipeline` / `WanAudioPipeline`.
# Changes from upstream: the model classes keep only the audio path (no image inputs, control adapters, state-dict
# converters or gradient checkpointing), `einops` is written out, attention always goes through
# `scaled_dot_product_attention`, the complex RoPE table is built per forward instead of being a buffer, `DAC` has no
# `audiotools` dependency (continuous VAE only, the checkpoint `.pth` is read by `from_pretrained`), the scheduler is a
# `SchedulerMixin` so diffusers can save and load it, the pipeline is a `DiffusionPipeline` over the same components so
# `RBLNDiffusionMixin` can compile them, and `encode_prompt` encodes one prompt at a time. The call signature and the
# numerics follow upstream.

import html
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import is_ftfy_available
from safetensors.torch import load_file
from torch import nn
from torch.nn import RMSNorm
from torch.nn.utils import weight_norm
from tqdm import tqdm


if is_ftfy_available():
    import ftfy


def _resolve_folder(pretrained_model_name_or_path: str | os.PathLike, subfolder: str | None, **kwargs: Any) -> Path:
    path = Path(pretrained_model_name_or_path)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                str(pretrained_model_name_or_path),
                allow_patterns=[f"{subfolder}/*"] if subfolder else None,
                cache_dir=kwargs.get("cache_dir"),
                revision=kwargs.get("revision"),
                token=kwargs.get("token"),
                local_files_only=kwargs.get("local_files_only", False),
            )
        )
    return path / subfolder if subfolder else path


# ----------------------------------------------------------------------------------------------------------------------
# WanAudioModel (diffsynth/models/wan_audio_dit.py, wan_video_dit.py)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 16384, theta: float = 10000.0, s: float = 1.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    pos = torch.arange(end, dtype=torch.float64, device=freqs.device) * s
    freqs = torch.outer(pos, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim, end, theta)
    return f_freqs_cis.chunk(3, dim=-1)


def rope_apply(x, freqs, num_heads):
    out_dtype = x.dtype
    b, s, _ = x.shape
    x = x.reshape(b, s, num_heads, -1)
    x = x.float().reshape(*x.shape[:-1], -1, 2)  # [b, s, n, d/2, 2]
    x_even, x_odd = x[..., 0], x[..., 1]
    cos = freqs.real.to(x_even.dtype)
    sin = freqs.imag.to(x_even.dtype)
    x_out = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(2)  # [b, s, n*d]
    return x_out.to(out_dtype)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int):
    q = q.reshape(q.shape[0], q.shape[1], num_heads, -1).transpose(1, 2)
    k = k.reshape(k.shape[0], k.shape[1], num_heads, -1).transpose(1, 2)
    v = v.reshape(v.shape[0], v.shape[1], num_heads, -1).transpose(1, 2)
    x = F.scaled_dot_product_attention(q, k, v)
    return x.transpose(1, 2).flatten(2)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = attention(q, k, v, self.num_heads)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = attention(q, k, v, self.num_heads)
        return self.o(x)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, context, t_mod, freqs):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(input_x, freqs)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, ...], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(1)).chunk(
            2, dim=1
        )
        return self.head(self.norm(x) * (1 + scale) + shift)


# Diffusers-style key names of the exported checkpoint -> the native names of `WanAudioModel`.
_HF_DIT_BLOCK_RENAME = {
    "attn1.norm_k.weight": "self_attn.norm_k.weight",
    "attn1.norm_q.weight": "self_attn.norm_q.weight",
    "attn1.to_k.bias": "self_attn.k.bias",
    "attn1.to_k.weight": "self_attn.k.weight",
    "attn1.to_out.0.bias": "self_attn.o.bias",
    "attn1.to_out.0.weight": "self_attn.o.weight",
    "attn1.to_q.bias": "self_attn.q.bias",
    "attn1.to_q.weight": "self_attn.q.weight",
    "attn1.to_v.bias": "self_attn.v.bias",
    "attn1.to_v.weight": "self_attn.v.weight",
    "attn2.norm_k.weight": "cross_attn.norm_k.weight",
    "attn2.norm_q.weight": "cross_attn.norm_q.weight",
    "attn2.to_k.bias": "cross_attn.k.bias",
    "attn2.to_k.weight": "cross_attn.k.weight",
    "attn2.to_out.0.bias": "cross_attn.o.bias",
    "attn2.to_out.0.weight": "cross_attn.o.weight",
    "attn2.to_q.bias": "cross_attn.q.bias",
    "attn2.to_q.weight": "cross_attn.q.weight",
    "attn2.to_v.bias": "cross_attn.v.bias",
    "attn2.to_v.weight": "cross_attn.v.weight",
    "ffn.net.0.proj.bias": "ffn.0.bias",
    "ffn.net.0.proj.weight": "ffn.0.weight",
    "ffn.net.2.bias": "ffn.2.bias",
    "ffn.net.2.weight": "ffn.2.weight",
    "norm2.bias": "norm3.bias",
    "norm2.weight": "norm3.weight",
    "scale_shift_table": "modulation",
}

_HF_DIT_GLOBAL_RENAME = {
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "scale_shift_table": "head.modulation",
    "proj_out.bias": "head.head.bias",
    "proj_out.weight": "head.head.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "patch_embedding.weight": "patch_embedding.weight",
}


def _convert_hf_dit_state_dict(state_dict: dict) -> dict:
    out = {}
    for key, param in state_dict.items():
        if key in _HF_DIT_GLOBAL_RENAME:
            out[_HF_DIT_GLOBAL_RENAME[key]] = param
        elif key.startswith("blocks."):
            parts = key.split(".", 2)
            block_idx, suffix = parts[1], parts[2]
            if suffix in _HF_DIT_BLOCK_RENAME:
                out[f"blocks.{block_idx}.{_HF_DIT_BLOCK_RENAME[suffix]}"] = param
            else:
                out[key] = param
        else:
            out[key] = param
    return out


class WanAudioModel(ModelMixin, ConfigMixin):
    """The MOSS-SoundEffect DiT: a stack of Wan blocks over 1-D audio latents with cross-attention to the text."""

    @register_to_config
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: tuple[int, ...],
        num_heads: int,
        num_layers: int,
        has_image_input: bool = False,
        vae_type: str = "dac",
    ):
        super().__init__()
        if has_image_input or vae_type != "dac":
            raise NotImplementedError("Only the text-to-audio WanAudioModel over the DAC latent space is ported.")
        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = tuple(patch_size)
        self.patch_embedding = nn.Conv1d(in_dim, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, ffn_dim, eps) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, self.patch_size, eps)
        self.head_dim = dim // num_heads

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        subfolder: str | None = None,
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> "WanAudioModel":
        folder = _resolve_folder(pretrained_model_name_or_path, subfolder, **kwargs)
        model = cls.from_config(cls.load_config(folder))
        state_dict = _convert_hf_dit_state_dict(load_file(folder / "diffusion_pytorch_model.safetensors"))
        model.load_state_dict(state_dict)
        if torch_dtype is not None:
            model = model.to(torch_dtype)
        return model.eval()

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        return x.transpose(1, 2).contiguous(), grid_size

    def unpatchify(self, x: torch.Tensor, grid_size):
        (f,) = grid_size
        (p,) = self.patch_size
        return x.view(x.shape[0], f, p, -1).permute(0, 3, 1, 2).reshape(x.shape[0], -1, f * p)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, **kwargs: Any):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        x, (f,) = self.patchify(x)
        # Upstream keeps the table in buffers; `model.to(dtype)` would discard their imaginary part.
        freqs = precompute_freqs_cis_1d(self.head_dim)
        freqs = (
            torch.cat(
                [
                    freqs[0][:f].view(f, -1).expand(f, -1),
                    freqs[1][:f].view(f, -1).expand(f, -1),
                    freqs[2][:f].view(f, -1).expand(f, -1),
                ],
                dim=-1,
            )
            .reshape(f, 1, -1)
            .to(x.device)
        )
        for block in self.blocks:
            x = block(x, context, t_mod, freqs)
        x = self.head(x, t)
        return self.unpatchify(x, (f,))


# ----------------------------------------------------------------------------------------------------------------------
# DAC (diffsynth/models/dac_vae.py), the continuous (VAE) variant


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self):
        return self.mean


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model: int = 64, strides: Sequence[int] = (2, 4, 8, 8), d_latent: int = 64):
        super().__init__()
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]
        self.block += [Snake1d(d_model), WNConv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, input_channel, channels, rates, d_out: int = 1):
        super().__init__()
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]
        layers += [Snake1d(output_dim), WNConv1d(output_dim, d_out, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DAC(ModelMixin, ConfigMixin):
    """The DAC audio VAE of MOSS-SoundEffect: a 1-D convolutional encoder and decoder around a Gaussian latent."""

    @register_to_config
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: Sequence[int] = (2, 4, 8, 8),
        latent_dim: int | None = None,
        decoder_dim: int = 1536,
        decoder_rates: Sequence[int] = (8, 8, 4, 2),
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int | list = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        continuous: bool = False,
    ):
        super().__init__()
        if not continuous:
            raise NotImplementedError("Only the continuous (VAE) DAC is ported; the quantized codec is not.")
        self.encoder_dim = encoder_dim
        self.encoder_rates = list(encoder_rates)
        self.decoder_dim = decoder_dim
        self.decoder_rates = list(decoder_rates)
        self.sample_rate = sample_rate
        self.continuous = continuous
        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = math.prod(self.encoder_rates)
        self.encoder = Encoder(encoder_dim, self.encoder_rates, latent_dim)
        self.quant_conv = nn.Conv1d(latent_dim, 2 * latent_dim, 1)
        self.post_quant_conv = nn.Conv1d(latent_dim, latent_dim, 1)
        self.decoder = Decoder(latent_dim, decoder_dim, self.decoder_rates)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        subfolder: str | None = None,
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> "DAC":
        folder = _resolve_folder(pretrained_model_name_or_path, subfolder, **kwargs)
        checkpoints = sorted(folder.glob("*.pth"))
        if checkpoints:
            # The upstream checkpoint: a state dict with the constructor arguments as metadata.
            artifacts = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
            model = cls(**artifacts["metadata"]["kwargs"])
            model.load_state_dict(artifacts["state_dict"])
        else:
            model = cls.from_config(cls.load_config(folder))
            model.load_state_dict(load_file(folder / "diffusion_pytorch_model.safetensors"))
        if torch_dtype is not None:
            model = model.to(torch_dtype)
        return model.eval()

    def encode(self, audio_data: torch.Tensor) -> DiagonalGaussianDistribution:
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(audio_data)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z))

    def forward(self, audio_data: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(audio_data).mode())


# ----------------------------------------------------------------------------------------------------------------------
# FlowMatchScheduler (diffsynth/schedulers/flow_match.py)


class FlowMatchScheduler(SchedulerMixin, ConfigMixin):
    order = 1

    @register_to_config
    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
        exponential_shift: bool = False,
        exponential_shift_mu: float | None = None,
        shift_terminal: float | None = None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, shift=None, dynamic_shift_len=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = (
                self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            )
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_ - sigma)

    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        dists = (self.timesteps[None, :] - timestep[:, None]).abs()
        timestep_ids = dists.argmin(dim=1)
        sigmas = self.sigmas[timestep_ids].to(original_samples.device).view(-1, 1, 1)
        return (1 - sigmas) * original_samples + sigmas * noise

    def calculate_shift(self, image_seq_len, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b


# ----------------------------------------------------------------------------------------------------------------------
# MossSoundEffectPipeline (pipeline_moss_soundeffect.py, diffsynth/pipelines/wan_audio.py, prompters/wan_prompter.py)


def basic_clean(text):
    if is_ftfy_available():
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    return " ".join(text.split())


@dataclass
class MossSoundEffectPipelineOutput:
    audios: torch.Tensor
    sample_rate: int
    prompts: list[str]


# The checkpoint's `model_index.json` lists its components as `[class, folder]` pairs; diffusers wants
# `[library, class]`. The classes are the ones in this module, the text encoder is the HF model.
_UPSTREAM_COMPONENTS = {
    "transformer": (__name__, "WanAudioModel"),
    "vae": (__name__, "DAC"),
    "scheduler": (__name__, "FlowMatchScheduler"),
    "text_encoder": ("transformers", "Qwen3ForCausalLM"),
    "tokenizer": ("transformers", "Qwen2Tokenizer"),
}


class MossSoundEffectPipeline(DiffusionPipeline):
    """
    Text-to-audio with MOSS-SoundEffect: a flow-matching DiT over DAC latents, conditioned on Qwen3 hidden states.

    The pipeline always denoises `max_inference_seconds` of audio and crops the waveform to `seconds`.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    # `WanPrompter(text_len=512)`: every prompt is padded or truncated to this many tokens.
    text_len = 512

    def __init__(
        self,
        transformer: WanAudioModel,
        vae: DAC,
        text_encoder,
        tokenizer,
        scheduler: FlowMatchScheduler,
        sample_rate: int = 48000,
        max_inference_seconds: int = 30,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler
        )
        self.register_to_config(sample_rate=int(sample_rate), max_inference_seconds=int(max_inference_seconds))

    @classmethod
    def download(cls, pretrained_model_name, **kwargs) -> str:
        # diffusers' own download rejects the component entries of the upstream `model_index.json`; fetch the
        # snapshot as upstream does and let `load_config` translate the entries.
        from huggingface_hub import snapshot_download

        return snapshot_download(
            str(pretrained_model_name),
            cache_dir=kwargs.get("cache_dir"),
            revision=kwargs.get("revision"),
            token=kwargs.get("token"),
            local_files_only=kwargs.get("local_files_only", False),
            force_download=kwargs.get("force_download", False),
            proxies=kwargs.get("proxies"),
        )

    @classmethod
    def load_config(cls, *args: Any, **kwargs: Any) -> Any:
        config = super().load_config(*args, **kwargs)
        config_dict = config[0] if isinstance(config, tuple) else config
        for name, (library, class_name) in _UPSTREAM_COMPONENTS.items():
            value = config_dict.get(name)
            if isinstance(value, list) and len(value) == 2 and value[1] == name:
                config_dict[name] = [library, class_name]
        return config

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def max_inference_seconds(self) -> int:
        return self.config.max_inference_seconds

    def encode_prompt(self, prompt: list[str], device: torch.device) -> torch.Tensor:
        """The last hidden states of the text encoder, [batch, text_len, dim], zero after each prompt's last token."""
        prompt = [whitespace_clean(basic_clean(text)) for text in prompt]
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_len,
            add_special_tokens=True,
        )
        ids, mask = tokens.input_ids.to(device), tokens.attention_mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        # One prompt at a time: the RBLN prefill cannot run a prompt with no valid token, whose row upstream zeroes.
        rows = []
        for i, length in enumerate(seq_lens.tolist()):
            if length == 0:
                rows.append(None)
                continue
            outputs = self.text_encoder(
                input_ids=ids[i : i + 1], attention_mask=mask[i : i + 1], output_hidden_states=True, use_cache=False
            )
            prompt_emb = outputs.hidden_states[-1][0]
            prompt_emb[length:] = 0
            rows.append(prompt_emb)
        zeros = torch.zeros(
            self.text_len, self.text_encoder.config.hidden_size, dtype=self.text_encoder.dtype, device=device
        )
        return torch.stack([zeros if row is None else row for row in rows])

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        seconds: float = 10.0,
        num_inference_steps: int = 100,
        cfg_scale: float = 4.0,
        sigma_shift: float = 5.0,
        seed: int = 0,
        negative_prompt: str | list[str] = "",
        append_duration_suffix: bool = True,
        num_channels: int = 1,
        max_inference_seconds: int | None = None,
        return_dict: bool = False,
        progress_bar_cmd=tqdm,
    ) -> torch.Tensor | MossSoundEffectPipelineOutput:
        """Run denoising and return a waveform of shape (batch, channels, samples)."""
        seconds = round(float(seconds), 1)
        if seconds <= 0:
            raise ValueError(f"seconds must be > 0, got {seconds}")
        full_seconds = int(max_inference_seconds or self.config.max_inference_seconds)
        if seconds > full_seconds:
            raise ValueError(f"seconds={seconds} exceeds max_inference_seconds={full_seconds}")

        def _format(p: str) -> str:
            p = p.strip()
            return f"{p} duration: {seconds:.1f}s" if append_duration_suffix else p

        prompts = [_format(p) for p in prompt] if isinstance(prompt, (list, tuple)) else [_format(prompt)]
        batch_size = len(prompts)
        if isinstance(negative_prompt, (list, tuple)):
            negative_prompts = list(negative_prompt)
        else:
            negative_prompts = [negative_prompt] * batch_size
        device = self._execution_device
        hop_length = self.vae.hop_length
        num_samples = self.config.sample_rate * full_seconds // hop_length * hop_length

        self.scheduler.set_timesteps(int(num_inference_steps), denoising_strength=1.0, shift=float(sigma_shift))
        context_posi = self.encode_prompt(prompts, device)
        context_nega = self.encode_prompt(negative_prompts, device) if cfg_scale != 1.0 else None

        # Upstream samples the noise in float32 and stores it at its intermediate dtype, the dtype the text encoder
        # was loaded with; the latents turn float32 after the first scheduler step.
        generator = torch.Generator("cpu").manual_seed(int(seed))
        shape = (batch_size, self.transformer.config.in_dim, num_samples // hop_length)
        latents = torch.randn(shape, generator=generator, dtype=torch.float32).to(self.text_encoder.dtype)

        transformer_dtype = self.transformer.dtype
        context_posi = context_posi.to(transformer_dtype)
        context_nega = context_nega.to(transformer_dtype) if context_nega is not None else None
        for i, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            t = timestep.unsqueeze(0).to(device)
            x = latents.to(transformer_dtype)
            noise_pred = self.transformer(x, t, context_posi).float()
            if cfg_scale != 1.0:
                noise_pred_nega = self.transformer(x, t, context_nega).float()
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred - noise_pred_nega)
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[i], latents)

        max_decode_bs = 8
        audio = torch.cat(
            [
                self.vae.decode(latents[start : start + max_decode_bs].to(self.vae.dtype))
                for start in range(0, latents.shape[0], max_decode_bs)
            ]
        )
        audio = audio[:, :, : int(self.config.sample_rate * seconds)]
        if return_dict:
            return MossSoundEffectPipelineOutput(audios=audio, sample_rate=self.config.sample_rate, prompts=prompts)
        return audio

    def save_audio(self, audio: torch.Tensor, output_path: str | Path, sample_rate: int | None = None) -> str:
        import torchaudio

        wav = audio.detach().cpu()
        if wav.ndim == 3:
            wav = wav[0]
        elif wav.ndim == 1:
            wav = wav.unsqueeze(0)
        output_path = str(Path(output_path).expanduser().resolve())
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_path, wav.to(torch.float32), int(sample_rate or self.config.sample_rate))
        return output_path
