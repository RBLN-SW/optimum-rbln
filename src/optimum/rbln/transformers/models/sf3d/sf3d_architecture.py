# Copyright 2025 Rebellions Inc. All rights reserved.
# Portions derived from Stability-AI/stable-fast-3d (MIT License).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SF3D model architecture for RBLN compilation.

DINOv2 vision backbone is imported from ``transformers`` and extended
with ada-norm modulation.  All other components (backbone transformer,
decoder, isosurface extraction, etc.) come from the ``stable-fast-3d``
repository and are included here so that ``optimum-rbln`` has **zero
runtime dependency** on that repository.
"""

import dataclasses
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_model

logger = logging.getLogger(__name__)

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
from transformers.models.dinov2.modeling_dinov2 import (
    Dinov2Encoder as _Dinov2Encoder,
    Dinov2Layer as _Dinov2Layer,
    Dinov2Model as _Dinov2Model,
)


# ---------------------------------------------------------------------------
# BaseModule helpers (from sf3d.models.utils)
# ---------------------------------------------------------------------------


def _parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    cfg_ = cfg.copy()
    field_names = {f.name for f in dataclasses.fields(fields)}
    for key in list(cfg_.keys()):
        if key not in field_names:
            cfg_.pop(key)
    return OmegaConf.merge(OmegaConf.structured(fields), cfg_)


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


# ---------------------------------------------------------------------------
# Modulation (ada-norm conditioning)
# ---------------------------------------------------------------------------


class Modulation(nn.Module):
    def __init__(self, embedding_dim: int, condition_dim: int, zero_init: bool = False, single_layer: bool = False):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear1 = nn.Identity() if single_layer else nn.Linear(condition_dim, condition_dim)
        self.linear2 = nn.Linear(condition_dim, embedding_dim * 2)
        if zero_init:
            nn.init.zeros_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        emb = self.linear2(self.silu(self.linear1(condition)))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# DINOv2 — thin subclasses adding ada-norm modulation to transformers DINOv2
# ---------------------------------------------------------------------------


class Dinov2Layer(_Dinov2Layer):
    """Extends transformers ``Dinov2Layer`` with optional ada-norm modulation."""

    def __init__(self, config: Dinov2Config):
        super().__init__(config)
        self.norm1_modulation = None
        self.norm2_modulation = None

    def register_ada_norm_modulation(self, norm1_mod, norm2_mod):
        self.norm1_modulation = norm1_mod
        self.norm2_modulation = norm2_mod

    def forward(self, hidden_states, modulation_cond=None):
        hidden_states_norm = self.norm1(hidden_states)
        if self.norm1_modulation is not None:
            hidden_states_norm = self.norm1_modulation(hidden_states_norm, modulation_cond)
        attn_out = self.layer_scale1(self.attention(hidden_states_norm))
        hidden_states = self.drop_path(attn_out) + hidden_states
        layer_output = self.norm2(hidden_states)
        if self.norm2_modulation is not None:
            layer_output = self.norm2_modulation(layer_output, modulation_cond)
        layer_output = self.layer_scale2(self.mlp(layer_output))
        layer_output = self.drop_path(layer_output) + hidden_states
        return (layer_output,)


class Dinov2Encoder(_Dinov2Encoder):
    """Extends transformers ``Dinov2Encoder`` to use modulation-aware layers."""

    def __init__(self, config: Dinov2Config):
        super().__init__(config)
        self.layer = nn.ModuleList([Dinov2Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, modulation_cond=None,
                output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, modulation_cond=modulation_cond)
            hidden_states = layer_outputs[0]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


class Dinov2Model(_Dinov2Model):
    """Extends transformers ``Dinov2Model`` with ada-norm modulation support."""

    def __init__(self, config: Dinov2Config):
        super().__init__(config)
        self.encoder = Dinov2Encoder(config)

    def forward(self, pixel_values=None, modulation_cond=None,
                output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(
            embedding_output, modulation_cond=modulation_cond,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
        )
        sequence_output = self.layernorm(encoder_outputs[0])
        pooled_output = sequence_output[:, 0, :]
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output,
                                          hidden_states=encoder_outputs.hidden_states)

    def set_gradient_checkpointing(self, value=False):
        self.encoder.gradient_checkpointing = value


# ---------------------------------------------------------------------------
# Camera Embedder
# ---------------------------------------------------------------------------


class LinearCameraEmbedder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 25
        out_channels: int = 768
        conditions: List[str] = field(default_factory=list)

    cfg: Config

    def configure(self):
        self.linear = nn.Linear(self.cfg.in_channels, self.cfg.out_channels)

    def forward(self, **kwargs):
        parts = []
        for name in self.cfg.conditions:
            c = kwargs[name]
            parts.append(c.view(*c.shape[:2], -1))
        cond = torch.cat(parts, dim=-1)
        return self.linear(cond)


# ---------------------------------------------------------------------------
# Image Tokenizer
# ---------------------------------------------------------------------------


class DINOV2SingleImageTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "facebook/dinov2-large"
        width: int = 512
        height: int = 512
        modulation_cond_dim: int = 768

    cfg: Config

    def configure(self):
        self.model = Dinov2Model.from_pretrained(self.cfg.pretrained_model_name_or_path)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.model.set_gradient_checkpointing(False)
        modulations = []
        for layer in self.model.encoder.layer:
            n1 = Modulation(self.model.config.hidden_size, self.cfg.modulation_cond_dim, zero_init=True, single_layer=True)
            n2 = Modulation(self.model.config.hidden_size, self.cfg.modulation_cond_dim, zero_init=True, single_layer=True)
            layer.register_ada_norm_modulation(n1, n2)
            modulations += [n1, n2]
        self.modulations = nn.ModuleList(modulations)
        self.register_buffer("image_mean", torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1), persistent=False)

    def forward(self, images, modulation_cond=None, **kwargs):
        packed = images.ndim == 4
        if packed:
            images = images.unsqueeze(1)
            if modulation_cond is not None:
                modulation_cond = modulation_cond.unsqueeze(1) if modulation_cond.ndim == 2 else modulation_cond
        B, Nv = images.shape[:2]
        images = (images - self.image_mean) / self.image_std
        out = self.model(
            rearrange(images, "B N C H W -> (B N) C H W"),
            modulation_cond=rearrange(modulation_cond, "B N Cc -> (B N) Cc") if modulation_cond is not None else None,
        )
        local_features = out.last_hidden_state.permute(0, 2, 1)
        local_features = rearrange(local_features, "(B N) Ct Nt -> B N Ct Nt", B=B)
        return local_features.squeeze(1) if packed else local_features


# ---------------------------------------------------------------------------
# Triplane Tokenizer
# ---------------------------------------------------------------------------


class TriplaneLearnablePositionalEmbedding(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        plane_size: int = 96
        num_channels: int = 1024

    cfg: Config

    def configure(self):
        self.embeddings = nn.Parameter(
            torch.randn(3, self.cfg.num_channels, self.cfg.plane_size, self.cfg.plane_size) / math.sqrt(self.cfg.num_channels)
        )

    def forward(self, batch_size: int):
        return rearrange(repeat(self.embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=batch_size), "B Np Ct Hp Wp -> B Ct (Np Hp Wp)")

    def detokenize(self, tokens):
        B, Ct, Nt = tokens.shape
        return rearrange(tokens, "B Ct (Np Hp Wp) -> B Np Ct Hp Wp", Np=3, Hp=self.cfg.plane_size, Wp=self.cfg.plane_size)


# ---------------------------------------------------------------------------
# Backbone — TwoStreamInterleaveTransformer
# ---------------------------------------------------------------------------


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states, scale=1.0):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class CrossAttention(nn.Module):
    def __init__(self, dim, kv_dim=None, num_heads=16, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        kv_dim = dim if not kv_dim else kv_dim
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        B, N_q, C = x_q.shape
        N_kv = x_kv.shape[1]
        q = self.wq(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop, scale=self.scale).permute(0, 2, 1, 3)
        return self.proj_drop(self.proj(x.reshape(B, N_q, C)))


class _FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.net = nn.ModuleList([GEGLU(dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)])

    def forward(self, x):
        for m in self.net:
            x = m(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, dim, kv_dim=None, num_heads=16, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, ff_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, kv_dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, kv_dim=kv_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim, dropout=ff_drop)

    def forward(self, z, x):
        z_n = self.norm1(z)
        z = z + self.attn1(z_n, z_n)
        z_n = self.norm2(z)
        z = z + self.attn2(z_n, x if x is not None else z_n)
        z = z + self.ff(self.norm3(z))
        return z


class FuseBlock(nn.Module):
    def __init__(self, dim_z, dim_x, num_heads=16, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, ff_drop=0.0, norm_x_input=True):
        super().__init__()
        self.norm_x_input = norm_x_input
        if norm_x_input:
            self.norm_x = nn.LayerNorm(dim_x)
        self.attn = CrossAttention(dim_z, kv_dim=dim_x, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm_z1 = nn.LayerNorm(dim_z)
        self.norm_z2 = nn.LayerNorm(dim_z)
        self.ff = _FeedForward(dim_z, dropout=ff_drop)

    def forward(self, z, x):
        z = z + self.attn(self.norm_z1(z), self.norm_x(x) if self.norm_x_input else x)
        z = z + self.ff(self.norm_z2(z))
        return z


class TwoStreamBlock(nn.Module):
    def __init__(self, dim_latent, dim_input, num_basic_blocks=4, num_heads=16, qkv_bias=False,
                 attn_drop=0.0, proj_drop=0.0, ff_drop=0.0, norm_x_input=True, dim_cross=None):
        super().__init__()
        self.fuse_block_in = FuseBlock(dim_latent, dim_input, num_heads=num_heads, qkv_bias=qkv_bias,
                                        attn_drop=attn_drop, proj_drop=proj_drop, ff_drop=ff_drop, norm_x_input=norm_x_input)
        self.transformer_block = nn.ModuleList([
            BasicBlock(dim_latent, kv_dim=dim_cross, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop, ff_drop=ff_drop)
            for _ in range(num_basic_blocks)
        ])
        self.fuse_block_out = FuseBlock(dim_input, dim_latent, num_heads=num_heads, qkv_bias=qkv_bias,
                                         attn_drop=attn_drop, proj_drop=proj_drop, ff_drop=ff_drop, norm_x_input=norm_x_input)

    def forward(self, latent, inp, cross_input):
        latent = self.fuse_block_in(latent, inp)
        for block in self.transformer_block:
            latent = block(latent, cross_input)
        inp = self.fuse_block_out(inp, latent)
        return latent, inp


class TwoStreamInterleaveTransformer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        num_attention_heads: int = 16
        attention_head_dim: int = 64
        raw_triplane_channels: int = 1024
        triplane_channels: int = 1024
        raw_image_channels: int = 1024
        num_latents: int = 1792
        num_blocks: int = 4
        num_basic_blocks: int = 3
        dropout: float = 0.0
        latent_init_std: float = 0.02
        norm_num_groups: int = 32
        attention_bias: bool = False
        norm_x_input: bool = False
        cross_attention_dim: int = 1024
        mix_latent: bool = True

    cfg: Config

    def configure(self):
        self.mix_latent = self.cfg.mix_latent
        self.num_attention_heads = self.cfg.num_attention_heads
        self.latent_dim = self.cfg.num_attention_heads * self.cfg.attention_head_dim
        if self.cfg.norm_num_groups > 0:
            self.norm_triplane = nn.GroupNorm(self.cfg.norm_num_groups, self.cfg.raw_triplane_channels, eps=1e-6, affine=True)
        else:
            self.norm_triplane = nn.LayerNorm(self.cfg.raw_triplane_channels)
        self.proj_triplane = nn.Linear(self.cfg.raw_triplane_channels, self.cfg.triplane_channels)
        if self.mix_latent:
            self.norm_image = nn.LayerNorm(self.cfg.raw_image_channels)
            self.proj_image = nn.Linear(self.cfg.raw_image_channels, self.latent_dim)
        self.norm_latent = nn.LayerNorm(self.latent_dim)
        self.proj_latent = nn.Linear(self.latent_dim, self.latent_dim)
        self.latent_init = nn.Parameter(torch.zeros(1, self.cfg.num_latents, self.latent_dim))
        nn.init.normal_(self.latent_init, std=self.cfg.latent_init_std)
        self.main_blocks = nn.ModuleList([
            TwoStreamBlock(
                self.latent_dim, self.cfg.triplane_channels,
                num_basic_blocks=self.cfg.num_basic_blocks, num_heads=self.cfg.num_attention_heads,
                qkv_bias=self.cfg.attention_bias, proj_drop=self.cfg.dropout, ff_drop=self.cfg.dropout,
                norm_x_input=self.cfg.norm_x_input, dim_cross=self.cfg.cross_attention_dim,
            ) for _ in range(self.cfg.num_blocks)
        ])
        self.proj_out = nn.Linear(self.cfg.triplane_channels, self.cfg.raw_triplane_channels)

    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        if isinstance(self.norm_triplane, nn.GroupNorm):
            triplane_tokens = self.norm_triplane(hidden_states).permute(0, 2, 1)
        else:
            triplane_tokens = self.norm_triplane(hidden_states.permute(0, 2, 1))
        triplane_tokens = self.proj_triplane(triplane_tokens)
        if self.mix_latent:
            image_tokens = self.proj_image(self.norm_image(encoder_hidden_states))
        init_latents = self.proj_latent(self.norm_latent(self.latent_init.expand(hidden_states.shape[0], -1, -1)))
        latent_tokens = torch.cat([image_tokens, init_latents], dim=1) if self.mix_latent else init_latents
        for block in self.main_blocks:
            latent_tokens, triplane_tokens = block(latent_tokens, triplane_tokens, encoder_hidden_states)
        return self.proj_out(triplane_tokens).permute(0, 2, 1).contiguous() + hidden_states


# ---------------------------------------------------------------------------
# Post-Processor & Decoder
# ---------------------------------------------------------------------------


class PixelShuffleUpsampleNetwork(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 1024
        out_channels: int = 40
        scale_factor: int = 4
        conv_layers: int = 4
        conv_kernel_size: int = 3

    cfg: Config

    def configure(self):
        layers: list = []
        out_ch = self.cfg.out_channels * self.cfg.scale_factor ** 2
        in_ch = self.cfg.in_channels
        for i in range(self.cfg.conv_layers):
            cur_out = in_ch if i != self.cfg.conv_layers - 1 else out_ch
            layers.append(nn.Conv2d(in_ch, cur_out, self.cfg.conv_kernel_size, padding=(self.cfg.conv_kernel_size - 1) // 2))
            if i != self.cfg.conv_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        layers.append(nn.PixelShuffle(self.cfg.scale_factor))
        self.upsample = nn.Sequential(*layers)

    def forward(self, triplanes):
        return rearrange(self.upsample(rearrange(triplanes, "B Np Ci Hp Wp -> (B Np) Ci Hp Wp", Np=3)), "(B Np) Co Hp Wp -> B Np Co Hp Wp", Np=3)


def _get_activation(name):
    if name is None or name in ("none", "linear", "identity"):
        return lambda x: x
    if name == "trunc_exp":
        return lambda x: torch.exp(x)
    if name == "sigmoid":
        return torch.sigmoid
    if name == "tanh":
        return torch.tanh
    if name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    if name == "normalize_channel_last":
        return lambda x: F.normalize(x, dim=-1, p=2, eps=1e-7)
    return getattr(F, name)


@dataclass
class HeadSpec:
    name: str
    out_channels: int
    n_hidden_layers: int
    output_activation: Optional[str] = None
    out_bias: float = 0.0


class MaterialMLP(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 120
        n_neurons: int = 64
        activation: str = "silu"
        heads: List[HeadSpec] = field(default_factory=list)

    cfg: Config

    def configure(self):
        heads = {}
        for head in self.cfg.heads:
            if isinstance(head, dict):
                head = HeadSpec(**head)
            layers: list = []
            for i in range(head.n_hidden_layers):
                layers += [nn.Linear(self.cfg.in_channels if i == 0 else self.cfg.n_neurons, self.cfg.n_neurons),
                           nn.SiLU(inplace=True) if self.cfg.activation == "silu" else nn.ReLU(inplace=True)]
            layers.append(nn.Linear(self.cfg.n_neurons, head.out_channels))
            heads[head.name] = nn.Sequential(*layers)
        self.heads = nn.ModuleDict(heads)

    def forward(self, x, include=None, exclude=None):
        heads = self.cfg.heads
        if include is not None:
            heads = [h for h in heads if (h.name if isinstance(h, HeadSpec) else h["name"]) in include]
        elif exclude is not None:
            heads = [h for h in heads if (h.name if isinstance(h, HeadSpec) else h["name"]) not in exclude]
        out = {}
        for h in heads:
            if isinstance(h, dict):
                h = HeadSpec(**h)
            out[h.name] = _get_activation(h.output_activation)(self.heads[h.name](x) + h.out_bias)
        return out


# ---------------------------------------------------------------------------
# Isosurface — MarchingTetrahedraHelper
# ---------------------------------------------------------------------------


def _generate_tetrahedral_grid(resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a regular tetrahedral grid via Freudenthal/Kuhn triangulation.

    Each voxel in the ``resolution^3`` grid is subdivided into 6 tetrahedra
    that share a consistent face connectivity across neighbouring cells.

    Returns ``(vertices, indices)`` where *vertices* is ``((N+1)^3, 3)``
    float32 in [-1, 1] and *indices* is ``(N^3 * 6, 4)`` int64.
    """
    n = resolution + 1
    lin = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    vertices = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    ci, cj, ck = np.meshgrid(
        np.arange(resolution, dtype=np.int64),
        np.arange(resolution, dtype=np.int64),
        np.arange(resolution, dtype=np.int64),
        indexing="ij",
    )
    ci, cj, ck = ci.ravel(), cj.ravel(), ck.ravel()

    v0 = ci * n * n + cj * n + ck
    v1 = v0 + n * n
    v2 = v0 + n
    v3 = v1 + n
    v4 = v0 + 1
    v5 = v1 + 1
    v6 = v2 + 1
    v7 = v3 + 1

    num_cubes = len(ci)
    indices = np.empty((num_cubes * 6, 4), dtype=np.int64)
    indices[0::6] = np.stack([v0, v1, v3, v7], axis=1)
    indices[1::6] = np.stack([v0, v1, v5, v7], axis=1)
    indices[2::6] = np.stack([v0, v2, v3, v7], axis=1)
    indices[3::6] = np.stack([v0, v2, v6, v7], axis=1)
    indices[4::6] = np.stack([v0, v4, v5, v7], axis=1)
    indices[5::6] = np.stack([v0, v4, v6, v7], axis=1)
    return vertices, indices


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (-1, 1)

    @property
    def grid_vertices(self):
        raise NotImplementedError


class MarchingTetrahedraHelper(IsosurfaceHelper):
    def __init__(self, resolution: int):
        super().__init__()
        self.resolution = resolution

        self.register_buffer("triangle_table", torch.as_tensor([
            [-1,-1,-1,-1,-1,-1],[1,0,2,-1,-1,-1],[4,0,3,-1,-1,-1],[1,4,2,1,3,4],
            [3,1,5,-1,-1,-1],[2,3,0,2,5,3],[1,4,0,1,5,4],[4,2,5,-1,-1,-1],
            [4,5,2,-1,-1,-1],[4,1,0,4,5,1],[3,2,0,3,5,2],[1,3,5,-1,-1,-1],
            [4,1,2,4,3,1],[3,0,4,-1,-1,-1],[2,0,1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],
        ], dtype=torch.long), persistent=False)
        self.register_buffer("num_triangles_table", torch.as_tensor(
            [0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long), persistent=False)
        self.register_buffer("base_tet_edges", torch.as_tensor(
            [0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long), persistent=False)

        tet_verts, tet_indices = _generate_tetrahedral_grid(resolution)
        self.register_buffer("_grid_vertices", torch.from_numpy(tet_verts).float(), persistent=False)
        self.register_buffer("indices", torch.from_numpy(tet_indices).long(), persistent=False)
        self._all_edges = None

        center_idx, boundary_idx = self._get_center_boundary(self._grid_vertices)
        self.register_buffer("center_indices", center_idx, persistent=False)
        self.register_buffer("boundary_indices", boundary_idx, persistent=False)

    @staticmethod
    def _get_center_boundary(verts):
        magn = torch.sum(verts ** 2, dim=-1)
        center_idx = torch.argmin(magn)
        boundary = (verts == verts.max()) | (verts == verts.min())
        boundary_idx = torch.nonzero(boundary.float().sum(-1)).squeeze(-1)
        return center_idx, boundary_idx

    @property
    def grid_vertices(self):
        return self._grid_vertices

    def normalize_grid_deformation(self, offsets):
        return (self.points_range[1] - self.points_range[0]) / self.resolution * torch.tanh(offsets)

    def _sort_edges(self, edges):
        with torch.no_grad():
            order = (edges[:, 0] > edges[:, 1]).long().unsqueeze(1)
            a = torch.gather(edges, 1, order)
            b = torch.gather(edges, 1, 1 - order)
        return torch.stack([a, b], -1)

    def _forward(self, pos, sdf, tets):
        with torch.no_grad():
            occ = sdf > 0
            occ_fx4 = occ[tets.reshape(-1)].reshape(-1, 4)
            occ_sum = occ_fx4.sum(-1)
            valid = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid]
            all_edges = tets[valid][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self._sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
            unique_edges = unique_edges.long()
            mask_edges = occ[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = torch.ones(unique_edges.shape[0], dtype=torch.long, device=pos.device) * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=pos.device)
            idx_map = mapping[idx_map]
            interp_v = unique_edges[mask_edges]

        edges_to_interp = pos[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1
        denom = edges_to_interp_sdf.sum(1, keepdim=True)
        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denom
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)
        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=pos.device))
        tetindex = (occ_fx4[valid] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]
        faces = torch.cat((
            torch.gather(idx_map[num_triangles == 1], 1, self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
            torch.gather(idx_map[num_triangles == 2], 1, self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
        ), dim=0)
        return verts, faces

    def forward(self, level, deformation=None):
        grid_verts = self.grid_vertices + self.normalize_grid_deformation(deformation) if deformation is not None else self.grid_vertices
        v_pos, t_pos_idx = self._forward(grid_verts, level, self.indices)
        return v_pos, t_pos_idx


# ---------------------------------------------------------------------------
# Estimators (needed for model loading only)
# ---------------------------------------------------------------------------


class ClipBasedHeadEstimator(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        model: str = "ViT-B-32"
        pretrain: str = "laion2b_s34b_b79k"
        distribution: str = "beta"
        distribution_eval: str = "mode"
        activation: str = "relu"
        hidden_features: int = 512
        heads: List[Dict] = field(default_factory=list)

    cfg: Config

    def configure(self):
        try:
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.cfg.model, pretrained=self.cfg.pretrain
            )
        except ImportError:
            logger.warning(
                "open_clip not installed; ClipBasedHeadEstimator weights will not be loaded. "
                "This is acceptable if estimators are not needed (weights loaded via strict=False)."
            )
            self.model = nn.Identity()
        for p in self.model.parameters():
            p.requires_grad_(False)

        act_fn = nn.ReLU if self.cfg.activation == "relu" else nn.SiLU
        heads = {}
        for h in self.cfg.heads:
            name = h["name"] if isinstance(h, dict) else h.name
            n_layers = h["n_hidden_layers"] if isinstance(h, dict) else h.n_hidden_layers
            hidden_layers: list = []
            for _ in range(n_layers):
                hidden_layers += [nn.Linear(self.cfg.hidden_features, self.cfg.hidden_features), act_fn(inplace=True)]
            head_items = [nn.Sequential(*hidden_layers)]
            for _ in range(2):
                head_items.append(nn.Sequential(
                    nn.Linear(self.cfg.hidden_features, self.cfg.hidden_features),
                    act_fn(inplace=True),
                    nn.Linear(self.cfg.hidden_features, 1),
                ))
            heads[name] = nn.ModuleList(head_items)
        self.heads = nn.ModuleDict(heads)

    def forward(self, cond_image):
        return {}


class MultiHeadEstimator(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        triplane_features: int = 1024
        n_layers: int = 2
        hidden_features: int = 512
        activation: str = "relu"
        pool: str = "max"
        heads: List[Dict] = field(default_factory=list)

    cfg: Config

    def configure(self):
        act_fn = nn.ReLU if self.cfg.activation == "relu" else nn.SiLU
        shared_layers: list = []
        cur_features = self.cfg.triplane_features * 3
        for _ in range(self.cfg.n_layers):
            shared_layers.append(nn.Conv2d(cur_features, self.cfg.hidden_features, kernel_size=3, padding=0, stride=2))
            shared_layers.append(act_fn(inplace=True))
            cur_features = self.cfg.hidden_features
        self.layers = nn.Sequential(*shared_layers)

        heads = {}
        for h in self.cfg.heads:
            name = h["name"] if isinstance(h, dict) else h.name
            out_ch = h["out_channels"] if isinstance(h, dict) else h.out_channels
            n_layers = h["n_hidden_layers"] if isinstance(h, dict) else h.n_hidden_layers
            head_layers: list = []
            for _ in range(n_layers):
                head_layers += [nn.Linear(self.cfg.hidden_features, self.cfg.hidden_features), act_fn(inplace=True)]
            head_layers.append(nn.Linear(self.cfg.hidden_features, out_ch))
            heads[name] = nn.Sequential(*head_layers)
        self.heads = nn.ModuleDict(heads)

    def forward(self, triplane_features):
        return {}


# ---------------------------------------------------------------------------
# SF3D main class
# ---------------------------------------------------------------------------

_SF3D_CLASS_REGISTRY: Dict[str, type] = {
    "sf3d.models.camera.LinearCameraEmbedder": LinearCameraEmbedder,
    "sf3d.models.tokenizers.image.DINOV2SingleImageTokenizer": DINOV2SingleImageTokenizer,
    "sf3d.models.tokenizers.triplane.TriplaneLearnablePositionalEmbedding": TriplaneLearnablePositionalEmbedding,
    "sf3d.models.transformers.backbone.TwoStreamInterleaveTransformer": TwoStreamInterleaveTransformer,
    "sf3d.models.network.PixelShuffleUpsampleNetwork": PixelShuffleUpsampleNetwork,
    "sf3d.models.network.MaterialMLP": MaterialMLP,
    "sf3d.models.image_estimator.clip_based_estimator.ClipBasedHeadEstimator": ClipBasedHeadEstimator,
    "sf3d.models.global_estimator.multi_head_estimator.MultiHeadEstimator": MultiHeadEstimator,
}


def _find_class(cls_string: str):
    if cls_string in _SF3D_CLASS_REGISTRY:
        return _SF3D_CLASS_REGISTRY[cls_string]
    raise ValueError(f"Unknown class: {cls_string}. Available: {list(_SF3D_CLASS_REGISTRY.keys())}")


class SF3D(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int = 512
        isosurface_resolution: int = 160
        isosurface_threshold: float = 10.0
        radius: float = 1.0
        background_color: list = field(default_factory=lambda: [0.5, 0.5, 0.5])
        default_fovy_deg: float = 40.0
        default_distance: float = 1.6
        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)
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
        image_estimator_cls: str = ""
        image_estimator: dict = field(default_factory=dict)
        global_estimator_cls: str = ""
        global_estimator: dict = field(default_factory=dict)

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
        load_model(model, weight_path, strict=False)
        return model

    def configure(self):
        self.image_tokenizer = _find_class(self.cfg.image_tokenizer_cls)(self.cfg.image_tokenizer)
        self.tokenizer = _find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.camera_embedder = _find_class(self.cfg.camera_embedder_cls)(self.cfg.camera_embedder)
        self.backbone = _find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = _find_class(self.cfg.post_processor_cls)(self.cfg.post_processor)
        self.decoder = _find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        if self.cfg.image_estimator_cls:
            self.image_estimator = _find_class(self.cfg.image_estimator_cls)(self.cfg.image_estimator)
        else:
            self.image_estimator = None
        if self.cfg.global_estimator_cls:
            self.global_estimator = _find_class(self.cfg.global_estimator_cls)(self.cfg.global_estimator)
        else:
            self.global_estimator = None
        self.isosurface_helper = MarchingTetrahedraHelper(self.cfg.isosurface_resolution)
