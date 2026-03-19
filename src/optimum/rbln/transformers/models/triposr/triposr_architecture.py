# Copyright 2025 Rebellions Inc. All rights reserved.
# Portions derived from VAST-AI-Research/TripoSR (MIT License).
# Portions derived from The HuggingFace Team (Apache License 2.0).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TripoSR model architecture — self-contained port for RBLN compilation.

All classes originally come from the ``TripoSR`` repository and the
Diffusers-style Attention layers.  They are included here so that
``optimum-rbln`` has **zero runtime dependency** on that repository.
"""

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from transformers.models.vit.modeling_vit import ViTModel


# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------


def _parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    return OmegaConf.merge(OmegaConf.structured(fields), cfg)


class BaseModule(nn.Module):
    @dataclass
    class Config:
        pass

    cfg: "Config"

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None, *args, **kwargs):
        super().__init__()
        self.cfg = _parse_structured(self.Config, cfg)
        self.configure(*args, **kwargs)

    def configure(self, *args, **kwargs) -> None:
        raise NotImplementedError


def scale_tensor(
    dat: torch.FloatTensor,
    inp_scale: Union[Tuple[float, float], torch.FloatTensor],
    tgt_scale: Union[Tuple[float, float], torch.FloatTensor],
) -> torch.FloatTensor:
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "softplus":
        return lambda x: F.softplus(x)
    else:
        return getattr(F, name)


def chunk_batch(func: Callable, chunk_size: int, *args, **kwargs) -> Any:
    if chunk_size <= 0:
        return func(*args, **kwargs)
    B = None
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    assert B is not None
    out = defaultdict(list)
    out_type = None
    for i in range(0, max(1, B), chunk_size):
        out_chunk = func(
            *[arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg for arg in args],
            **{k: arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg for k, arg in kwargs.items()},
        )
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, (tuple, list)):
            chunk_length = len(out_chunk)
            out_chunk = {i: c for i, c in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            raise TypeError(f"Unsupported return type: {type(out_chunk)}")
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            out[k].append(v)
    if out_type is None:
        return None
    out_merged: Dict[Any, Optional[torch.Tensor]] = {}
    for k, v in out.items():
        if all(vv is None for vv in v):
            out_merged[k] = None
        elif all(isinstance(vv, torch.Tensor) for vv in v):
            out_merged[k] = torch.cat(v, dim=0)
        else:
            raise TypeError(f"Mixed types in chunk output for key {k}")
    if out_type is torch.Tensor:
        return out_merged[0]
    elif out_type in (tuple, list):
        return out_type([out_merged[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out_merged


class ImagePreprocessor:
    def convert_and_resize(self, image: Union[PIL.Image.Image, np.ndarray, torch.Tensor], size: int):
        if isinstance(image, PIL.Image.Image):
            image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = torch.from_numpy(image.astype(np.float32) / 255.0)
            else:
                image = torch.from_numpy(image)
        batched = image.ndim == 4
        if not batched:
            image = image[None, ...]
        image = F.interpolate(image.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1)
        if not batched:
            image = image[0]
        return image

    def __call__(self, image, size: int) -> Any:
        if isinstance(image, (np.ndarray, torch.FloatTensor)) and image.ndim == 4:
            image = self.convert_and_resize(image, size)
        else:
            if not isinstance(image, list):
                image = [image]
            image = [self.convert_and_resize(im, size) for im in image]
            image = torch.stack(image, dim=0)
        return image


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor=None,
        out_dim: int = None,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self._from_deprecated_attn_block = _from_deprecated_attn_block
        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        self.spatial_norm = None

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            norm_cross_ch = added_kv_proj_dim if self.added_kv_proj_dim is not None else self.cross_attention_dim
            self.norm_cross = nn.GroupNorm(num_channels=norm_cross_ch, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True)
        else:
            raise ValueError(f"unknown cross_attention_norm: {cross_attention_norm}")

        self.linear_cls = nn.Linear
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        if not self.only_cross_attention:
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)

        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, self.out_dim, bias=out_bias), nn.Dropout(dropout)])

        if processor is None:
            processor = AttnProcessor2_0()
        self.processor = processor

    def set_processor(self, processor):
        self.processor = processor

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        return self.processor(self, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **cross_attention_kwargs)

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        return tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)

    def head_to_batch_dim(self, tensor, out_dim=3):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size).permute(0, 2, 1, 3)
        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def prepare_attention_mask(self, attention_mask, target_length, batch_size, out_dim=3):
        head_size = self.heads
        if attention_mask is None:
            return attention_mask
        current_length = attention_mask.shape[-1]
        if current_length != target_length:
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1).repeat_interleave(head_size, dim=1)
        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states):
        if isinstance(self.norm_cross, nn.LayerNorm):
            return self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            return self.norm_cross(encoder_hidden_states.transpose(1, 2)).transpose(1, 2)
        return encoder_hidden_states


class AttnProcessor2_0:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# Feed-forward & activations
# ---------------------------------------------------------------------------


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def forward(self, hidden_states):
        return F.gelu(self.proj(hidden_states), approximate=self.approximate)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states, scale: float = 1.0):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class ApproximateGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return self.proj(x) * torch.sigmoid(1.702 * self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4,
                 dropout: float = 0.0, activation_fn: str = "geglu", final_dropout: bool = False):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)
        else:
            raise ValueError(f"Unknown activation_fn: {activation_fn}")
        self.net = nn.ModuleList([act_fn, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)])
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# BasicTransformerBlock
# ---------------------------------------------------------------------------


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int,
                 dropout=0.0, cross_attention_dim: Optional[int] = None, activation_fn: str = "geglu",
                 attention_bias: bool = False, only_cross_attention: bool = False,
                 double_self_attention: bool = False, upcast_attention: bool = False,
                 norm_elementwise_affine: bool = True, norm_type: str = "layer_norm", final_dropout: bool = False):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout,
            bias=attention_bias, cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            self.attn2 = Attention(
                query_dim=dim, cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout,
                bias=attention_bias, upcast_attention=upcast_attention,
            )
        else:
            self.norm2 = None
            self.attn2 = None
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
        )
        hidden_states = attn_output + hidden_states
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=encoder_attention_mask,
            )
            hidden_states = attn_output + hidden_states
        norm_hidden_states = self.norm3(hidden_states)
        if self._chunk_size is not None:
            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat([self.ff(hid_slice) for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)], dim=self._chunk_dim)
        else:
            ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Transformer1D backbone
# ---------------------------------------------------------------------------


class Transformer1D(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        num_attention_heads: int = 16
        attention_head_dim: int = 88
        in_channels: Optional[int] = None
        out_channels: Optional[int] = None
        num_layers: int = 1
        dropout: float = 0.0
        norm_num_groups: int = 32
        cross_attention_dim: Optional[int] = None
        attention_bias: bool = False
        activation_fn: str = "geglu"
        only_cross_attention: bool = False
        double_self_attention: bool = False
        upcast_attention: bool = False
        norm_type: str = "layer_norm"
        norm_elementwise_affine: bool = True
        gradient_checkpointing: bool = False

    cfg: Config

    def configure(self) -> None:
        self.num_attention_heads = self.cfg.num_attention_heads
        self.attention_head_dim = self.cfg.attention_head_dim
        inner_dim = self.num_attention_heads * self.attention_head_dim
        self.in_channels = self.cfg.in_channels
        self.norm = nn.GroupNorm(num_groups=self.cfg.norm_num_groups, num_channels=self.cfg.in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(self.cfg.in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim, self.num_attention_heads, self.attention_head_dim,
                dropout=self.cfg.dropout, cross_attention_dim=self.cfg.cross_attention_dim,
                activation_fn=self.cfg.activation_fn, attention_bias=self.cfg.attention_bias,
                only_cross_attention=self.cfg.only_cross_attention, double_self_attention=self.cfg.double_self_attention,
                upcast_attention=self.cfg.upcast_attention, norm_type=self.cfg.norm_type,
                norm_elementwise_affine=self.cfg.norm_elementwise_affine,
            ) for _ in range(self.cfg.num_layers)
        ])
        self.out_channels = self.cfg.in_channels if self.cfg.out_channels is None else self.cfg.out_channels
        self.proj_out = nn.Linear(inner_dim, self.cfg.in_channels)
        self.gradient_checkpointing = self.cfg.gradient_checkpointing

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, encoder_attention_mask=None):
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        batch, _, seq_len = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 1).reshape(batch, seq_len, inner_dim)
        hidden_states = self.proj_in(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask,
                                  encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, seq_len, inner_dim).permute(0, 2, 1).contiguous()
        return hidden_states + residual


# ---------------------------------------------------------------------------
# Image tokenizer
# ---------------------------------------------------------------------------


class DINOSingleImageTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "facebook/dino-vitb16"
        enable_gradient_checkpointing: bool = False

    cfg: Config

    def configure(self) -> None:
        self.model: ViTModel = ViTModel(
            ViTModel.config_class.from_pretrained(
                hf_hub_download(repo_id=self.cfg.pretrained_model_name_or_path, filename="config.json")
            )
        )
        if self.cfg.enable_gradient_checkpointing:
            self.model.encoder.gradient_checkpointing = True
        self.register_buffer("image_mean", torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1), persistent=False)

    def forward(self, images, **kwargs):
        packed = False
        if images.ndim == 4:
            packed = True
            images = images.unsqueeze(1)
        batch_size, n_input_views = images.shape[:2]
        images = (images - self.image_mean) / self.image_std
        out = self.model(rearrange(images, "B N C H W -> (B N) C H W"), interpolate_pos_encoding=True)
        local_features = out.last_hidden_state.permute(0, 2, 1)
        local_features = rearrange(local_features, "(B N) Ct Nt -> B N Ct Nt", B=batch_size)
        if packed:
            local_features = local_features.squeeze(1)
        return local_features


# ---------------------------------------------------------------------------
# Triplane tokenizer
# ---------------------------------------------------------------------------


class Triplane1DTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        plane_size: int = 0
        num_channels: int = 0

    cfg: Config

    def configure(self) -> None:
        self.embeddings = nn.Parameter(
            torch.randn(3, self.cfg.num_channels, self.cfg.plane_size, self.cfg.plane_size) / math.sqrt(self.cfg.num_channels)
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        return rearrange(repeat(self.embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=batch_size), "B Np Ct Hp Wp -> B Ct (Np Hp Wp)")

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, Ct, Nt = tokens.shape
        return rearrange(tokens, "B Ct (Np Hp Wp) -> B Np Ct Hp Wp", Np=3, Hp=self.cfg.plane_size, Wp=self.cfg.plane_size)


# ---------------------------------------------------------------------------
# Post-processor and decoder
# ---------------------------------------------------------------------------


class TriplaneUpsampleNetwork(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 0
        out_channels: int = 0

    cfg: Config

    def configure(self) -> None:
        self.upsample = nn.ConvTranspose2d(self.cfg.in_channels, self.cfg.out_channels, kernel_size=2, stride=2)

    def forward(self, triplanes: torch.Tensor) -> torch.Tensor:
        return rearrange(
            self.upsample(rearrange(triplanes, "B Np Ci Hp Wp -> (B Np) Ci Hp Wp", Np=3)),
            "(B Np) Co Hp Wp -> B Np Co Hp Wp", Np=3,
        )


class NeRFMLP(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 0
        n_neurons: int = 0
        n_hidden_layers: int = 0
        activation: str = "relu"
        bias: bool = True
        weight_init: Optional[str] = "kaiming_uniform"
        bias_init: Optional[str] = None

    cfg: Config

    def configure(self) -> None:
        layers = [self._make_linear(self.cfg.in_channels, self.cfg.n_neurons), self._make_activation()]
        for _ in range(self.cfg.n_hidden_layers - 1):
            layers += [self._make_linear(self.cfg.n_neurons, self.cfg.n_neurons), self._make_activation()]
        layers.append(self._make_linear(self.cfg.n_neurons, 4))
        self.layers = nn.Sequential(*layers)

    def _make_linear(self, dim_in, dim_out):
        layer = nn.Linear(dim_in, dim_out, bias=self.cfg.bias)
        if self.cfg.weight_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        if self.cfg.bias and self.cfg.bias_init == "zero":
            nn.init.zeros_(layer.bias)
        return layer

    def _make_activation(self):
        if self.cfg.activation == "relu":
            return nn.ReLU(inplace=True)
        elif self.cfg.activation == "silu":
            return nn.SiLU(inplace=True)
        raise NotImplementedError(f"Unsupported activation: {self.cfg.activation}")

    def forward(self, x):
        inp_shape = x.shape[:-1]
        features = self.layers(x.reshape(-1, x.shape[-1])).reshape(*inp_shape, -1)
        return {"density": features[..., 0:1], "features": features[..., 1:4]}


# ---------------------------------------------------------------------------
# NeRF Renderer
# ---------------------------------------------------------------------------


class TriplaneNeRFRenderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        radius: float = 0.5
        feature_reduction: str = "concat"
        density_activation: str = "trunc_exp"
        density_bias: float = -1.0
        color_activation: str = "sigmoid"
        num_samples_per_ray: int = 128
        randomized: bool = False

    cfg: Config

    def configure(self) -> None:
        self.chunk_size = 0

    def set_chunk_size(self, chunk_size: int):
        self.chunk_size = chunk_size

    def query_triplane(self, decoder, positions, triplane):
        input_shape = positions.shape[:-1]
        positions = positions.view(-1, 3)
        positions = scale_tensor(positions, (-self.cfg.radius, self.cfg.radius), (-1, 1))

        def _query_chunk(x):
            indices2D = torch.stack((x[..., [0, 1]], x[..., [0, 2]], x[..., [1, 2]]), dim=-3)
            out = F.grid_sample(
                rearrange(triplane, "Np Cp Hp Wp -> Np Cp Hp Wp", Np=3),
                rearrange(indices2D, "Np N Nd -> Np () N Nd", Np=3),
                align_corners=False, mode="bilinear",
            )
            if self.cfg.feature_reduction == "concat":
                out = rearrange(out, "Np Cp () N -> N (Np Cp)", Np=3)
            elif self.cfg.feature_reduction == "mean":
                out = reduce(out, "Np Cp () N -> N Cp", Np=3, reduction="mean")
            return decoder(out)

        if self.chunk_size > 0:
            net_out = chunk_batch(_query_chunk, self.chunk_size, positions)
        else:
            net_out = _query_chunk(positions)
        net_out["density_act"] = get_activation(self.cfg.density_activation)(net_out["density"] + self.cfg.density_bias)
        net_out["color"] = get_activation(self.cfg.color_activation)(net_out["features"])
        return {k: v.view(*input_shape, -1) for k, v in net_out.items()}

    def forward(self, decoder, triplane, rays_o, rays_d):
        return self._forward(decoder, triplane, rays_o, rays_d)

    def _forward(self, decoder, triplane, rays_o, rays_d):
        pass

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()


# ---------------------------------------------------------------------------
# Isosurface (skimage-based, replaces torchmcubes)
# ---------------------------------------------------------------------------


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self):
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            x, y, z = (torch.linspace(*self.points_range, self.resolution),) * 3
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            self._grid_vertices = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)
        return self._grid_vertices

    def forward(self, level: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        from skimage.measure import marching_cubes as sk_marching_cubes

        level_3d = -level.view(self.resolution, self.resolution, self.resolution).detach().cpu().numpy()
        verts_np, faces_np, _, _ = sk_marching_cubes(level_3d, level=0.0)
        v_pos = torch.from_numpy(verts_np[..., [2, 1, 0]].copy()).float() / (self.resolution - 1.0)
        t_pos_idx = torch.from_numpy(faces_np.copy()).long()
        return v_pos.to(level.device), t_pos_idx.to(level.device)


# ---------------------------------------------------------------------------
# TSR main class
# ---------------------------------------------------------------------------


_TSR_CLASS_REGISTRY: Dict[str, type] = {
    "tsr.models.tokenizers.image.DINOSingleImageTokenizer": DINOSingleImageTokenizer,
    "tsr.models.tokenizers.triplane.Triplane1DTokenizer": Triplane1DTokenizer,
    "tsr.models.transformer.transformer_1d.Transformer1D": Transformer1D,
    "tsr.models.network_utils.TriplaneUpsampleNetwork": TriplaneUpsampleNetwork,
    "tsr.models.network_utils.NeRFMLP": NeRFMLP,
    "tsr.models.nerf_renderer.TriplaneNeRFRenderer": TriplaneNeRFRenderer,
}


def _find_class(cls_string: str):
    if cls_string in _TSR_CLASS_REGISTRY:
        return _TSR_CLASS_REGISTRY[cls_string]
    raise ValueError(f"Unknown class: {cls_string}. Available: {list(_TSR_CLASS_REGISTRY.keys())}")


class TSR(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int = 0
        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)
        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)
        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)
        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)
        decoder_cls: str = ""
        decoder: dict = field(default_factory=dict)
        renderer_cls: str = ""
        renderer: dict = field(default_factory=dict)

    cfg: Config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, config_name: str, weight_name: str):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
        else:
            config_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename=config_name)
            weight_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename=weight_name)
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        ckpt = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(ckpt)
        return model

    def configure(self):
        self.image_tokenizer = _find_class(self.cfg.image_tokenizer_cls)(self.cfg.image_tokenizer)
        self.tokenizer = _find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = _find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = _find_class(self.cfg.post_processor_cls)(self.cfg.post_processor)
        self.decoder = _find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.renderer = _find_class(self.cfg.renderer_cls)(self.cfg.renderer)
        self.image_processor = ImagePreprocessor()
        self.isosurface_helper = None
