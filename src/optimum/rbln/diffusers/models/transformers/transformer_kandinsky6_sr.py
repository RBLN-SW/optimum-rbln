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

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import rebel
import torch
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_kandinsky6_sr import (
    FRACTAL_BLOCK_SIZE,
    Kandinsky6SRAttention,
    Kandinsky6SRRoPE3D,
    Kandinsky6SRTransformer3DModel,
    Kandinsky6SRTransformerBlock,
    _local_merge,
    _local_patch,
    sliding_tile_mask,
)

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ...configurations import RBLNKandinsky6SRTransformer3DModelConfig


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


NABLA_BLOCK_TOKENS = FRACTAL_BLOCK_SIZE**2
# Bisection steps of the top-cdf threshold. The softmax map is bfloat16, whose neighbouring values are at
# least 2**-8 of the cutoff apart, so 32 halvings of [0, max] isolate the cutoff while max / cutoff < 2**24.
NABLA_THRESHOLD_SEARCH_STEPS = 32
# The device runs the block-map ops only on axes of whole 64-element rows, so the map is padded to them.
NABLA_BLOCK_MAP_ALIGNMENT = 64


def nabla_aligned_blocks(num_blocks: int) -> int:
    return -(-num_blocks // NABLA_BLOCK_MAP_ALIGNMENT) * NABLA_BLOCK_MAP_ALIGNMENT


def nabla_top_cdf_cutoff(probs: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    The value below which NABLA drops blocks from the softmax map `probs` of shape (..., key blocks, query blocks).

    NABLA sorts the probabilities of a query block ascending and keeps the entries where the running sum reaches
    `1 - threshold`. Those are the entries above the largest value whose mass `sum(p for p in column if p <= value)`
    stays below `1 - threshold`, so that value is found by bisection on the monotone mass instead of sorting. The
    query blocks stay on the last axis, where the device keeps the per-column bounds.
    """
    target = 1.0 - threshold
    lower = torch.zeros_like(probs[..., :1, :])
    upper = probs.amax(dim=-2, keepdim=True)
    for _ in range(NABLA_THRESHOLD_SEARCH_STEPS):
        middle = (lower + upper) * 0.5
        reached = torch.where(probs <= middle, probs, 0.0).sum(dim=-2, keepdim=True) >= target
        upper = torch.where(reached, middle, upper)
        lower = torch.where(reached, lower, middle)
    return lower


def nabla_block_sparse_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, sta_mask: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    NABLA attention of `Kandinsky6SRAttnProcessor` on (batch, heads, tokens, head_dim) tensors.

    diffusers builds the block mask from the block-mean query/key statistics and runs flex attention on it. Here
    the same mask selects whole key blocks in a masked SDPA: the 64 tokens of a query block go on the group axis
    so that the mask of a query block is one row of key tokens.

    The block map is padded to `sta_mask`, which comes padded to whole 64-block rows. The padded keys get no
    probability and are outside the prior, so they are never kept; the rows of the padded queries are dropped.
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    num_blocks = seq_len // NABLA_BLOCK_TOKENS
    aligned_blocks = sta_mask.shape[-1]
    query_blocks = query.reshape(batch_size, num_heads, num_blocks, NABLA_BLOCK_TOKENS, head_dim)
    key_blocks = key.reshape(batch_size, num_heads, num_blocks, NABLA_BLOCK_TOKENS, head_dim)

    block_padding = (0, 0, 0, aligned_blocks - num_blocks)
    query_means = F.pad(query_blocks.mean(-2), block_padding)
    key_means = F.pad(key_blocks.mean(-2), block_padding)
    block_map = query_means @ key_means.transpose(-2, -1) / math.sqrt(head_dim)
    padded_keys = torch.arange(aligned_blocks) >= num_blocks
    block_map = torch.where(padded_keys, float("-inf"), block_map)
    probs = torch.softmax(block_map, dim=-1).float().transpose(-2, -1)
    cutoff = nabla_top_cdf_cutoff(probs, threshold)
    # Every probability is at most 1, so 2 keeps the blocks of the sliding-tile prior whatever the cutoff.
    block_mask = (torch.where(sta_mask.transpose(-2, -1), 2.0, probs) > cutoff).transpose(-2, -1)

    token_mask = block_mask.unsqueeze(-1).expand(-1, -1, -1, -1, NABLA_BLOCK_TOKENS).flatten(-2)
    token_mask = token_mask[:, :, :num_blocks, :seq_len]
    hidden_states = F.scaled_dot_product_attention(
        query_blocks.transpose(2, 3), key.unsqueeze(2), value.unsqueeze(2), attn_mask=token_mask.unsqueeze(2)
    )
    return hidden_states.transpose(2, 3).reshape(batch_size, num_heads, seq_len, head_dim)


def _rms_norm(hidden_states: torch.Tensor, norm: torch.nn.RMSNorm) -> torch.Tensor:
    # torch.rms_norm on a float32 input with the checkpoint-dtype weight of `Kandinsky6SRAttention`.
    eps = norm.eps if norm.eps is not None else torch.finfo(hidden_states.dtype).eps
    normalized = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + eps)
    return normalized * norm.weight.float()


def _pair_swap_matrix(head_dim: int, dtype: torch.dtype) -> torch.Tensor:
    """`x @ swap` maps every rotary pair (x0, x1) of `x` to (-x1, x0)."""
    swap = torch.zeros(head_dim, head_dim, dtype=dtype)
    even = torch.arange(0, head_dim, 2)
    swap[even + 1, even] = -1
    swap[even, even + 1] = 1
    return swap


def _apply_rotary(
    hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pair_swap: torch.Tensor
) -> torch.Tensor:
    # `apply_rotary` of diffusers. The pairs are swapped with a +-1 matmul, which is exact, instead of being split
    # off the last axis, so the float32 products and sums are those of diffusers.
    swapped = hidden_states @ pair_swap
    return (hidden_states.float() * cos + swapped.float() * sin).to(hidden_states.dtype)


class _Kandinsky6SRTimestepEmbedding(torch.nn.Module):
    """The timestep-only embedding and AdaLN projections, computed once per call on the device."""

    def __init__(self, model: Kandinsky6SRTransformer3DModel):
        super().__init__()
        self.time_embeddings = model.time_embeddings
        self.pooled_bias = model.pooled_bias
        self.visual_modulations = torch.nn.ModuleList(
            [block.visual_modulation for block in model.visual_transformer_blocks]
        )
        self.out_modulation = model.out_layer.modulation

    def forward(self, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temb = self.time_embeddings(timestep) + self.pooled_bias
        modulation = torch.stack([layer(temb) for layer in self.visual_modulations], dim=1).unflatten(-1, (6, -1))
        out_modulation = self.out_modulation(temb).unflatten(-1, (2, -1))
        return modulation.float(), out_modulation.float()


class Kandinsky6SRTransformer3DModelWrapper(torch.nn.Module):
    """
    The transformer blocks and the output projection of `Kandinsky6SRTransformer3DModel.forward`.

    The tokens arrive patchified and in NABLA block order (8x8 spatial neighbourhoods). The timestep-only AdaLN
    modulations come from the separate device graph, keeping their lifetimes independent of the token activations.
    """

    def __init__(self, model: Kandinsky6SRTransformer3DModel) -> None:
        super().__init__()
        self.visual_embeddings = model.visual_embeddings
        self.blocks = model.visual_transformer_blocks
        self.out_layer = model.out_layer
        self.threshold = model.config.nabla_threshold
        dtype = self.visual_embeddings.in_layer.weight.dtype
        self.register_buffer("pair_swap", _pair_swap_matrix(sum(model.config.axes_dims), dtype), persistent=False)

    def _attention(
        self,
        attn: Kandinsky6SRAttention,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        sta_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = attn.to_query(hidden_states).unflatten(-1, (attn.num_heads, -1))
        key = attn.to_key(hidden_states).unflatten(-1, (attn.num_heads, -1))
        value = attn.to_value(hidden_states).unflatten(-1, (attn.num_heads, -1))
        query = _rms_norm(query.float(), attn.query_norm).type_as(query)
        key = _rms_norm(key.float(), attn.key_norm).type_as(key)
        query = _apply_rotary(query, rope_cos, rope_sin, self.pair_swap)
        key = _apply_rotary(key, rope_cos, rope_sin, self.pair_swap)

        hidden_states = nabla_block_sparse_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), sta_mask, self.threshold
        )
        return attn.out_layer(hidden_states.transpose(1, 2).flatten(2, 3))

    def _block(
        self,
        block: Kandinsky6SRTransformerBlock,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        sta_mask: torch.Tensor,
    ) -> torch.Tensor:
        # `apply_scale_shift` and `apply_gate_sum` of diffusers, in float32 like there.
        shift, scale, gate = modulation[:, 0:1], modulation[:, 1:2], modulation[:, 2:3]
        normed = block.self_attention_norm(hidden_states.float()) * (scale + 1.0) + shift
        attn_output = self._attention(
            block.self_attention, normed.to(hidden_states.dtype), rope_cos, rope_sin, sta_mask
        )
        hidden_states = (hidden_states.float() + gate * attn_output.float()).to(hidden_states.dtype)

        shift, scale, gate = modulation[:, 3:4], modulation[:, 4:5], modulation[:, 5:6]
        normed = block.feed_forward_norm(hidden_states.float()) * (scale + 1.0) + shift
        ff_output = block.feed_forward(normed.to(hidden_states.dtype))
        return (hidden_states.float() + gate * ff_output.float()).to(hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        out_modulation: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        sta_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.visual_embeddings.in_layer(hidden_states)
        for index, block in enumerate(self.blocks):
            hidden_states = self._block(block, hidden_states, modulation[:, index], rope_cos, rope_sin, sta_mask)

        shift, scale = out_modulation[:, 0:1], out_modulation[:, 1:2]
        normed = self.out_layer.norm(hidden_states.float()) * (scale + 1.0) + shift
        return self.out_layer.out_layer(normed.to(hidden_states.dtype))


class RBLNKandinsky6SRTransformer3DModel(RBLNModel):
    """
    RBLN implementation of Kandinsky6SRTransformer3DModel for Kandinsky 6 video super-resolution.

    Two device graphs run the timestep embedding / AdaLN modulations and the transformer blocks with NABLA
    attention. Their separate activation lifetimes bound device memory. Patchifying and token order run on the host.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = Kandinsky6SRTransformer3DModel
    _rbln_config_class = RBLNKandinsky6SRTransformer3DModelConfig
    _output_class = Transformer2DModelOutput

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        config = self.config
        self.grid_size = self._grid_size(config, self.rbln_config)
        self.rope_cos, self.rope_sin = self._rotary_tables(config, self.grid_size)
        num_frames, height, width = self.grid_size
        sta_mask = sliding_tile_mask(
            num_frames,
            height // FRACTAL_BLOCK_SIZE,
            width // FRACTAL_BLOCK_SIZE,
            tuple(config.nabla_window),
            torch.device("cpu"),
        )
        padding = nabla_aligned_blocks(sta_mask.shape[-1]) - sta_mask.shape[-1]
        self.sta_mask = F.pad(sta_mask, (0, padding, 0, padding), value=False)

    @staticmethod
    def _grid_size(model_config, rbln_config: RBLNKandinsky6SRTransformer3DModelConfig) -> tuple[int, int, int]:
        """The (frames, height, width) token grid of a tile after `patch_size`."""
        patch_t, patch_h, patch_w = model_config.patch_size
        latent_height, latent_width = rbln_config.sample_size
        return rbln_config.num_latent_frames // patch_t, latent_height // patch_h, latent_width // patch_w

    @staticmethod
    def _rotary_tables(model_config, grid_size: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The RoPE of `Kandinsky6SRTransformer3DModel.forward` in NABLA block order, as (tokens, 1, head_dim) cos and
        sin tables that repeat every frequency for both members of its rotary pair.
        """
        rope = Kandinsky6SRRoPE3D(tuple(model_config.axes_dims))
        positions = tuple(torch.arange(size) for size in grid_size)
        rotary = rope(positions, tuple(model_config.scale_factor))
        group_size = (1, FRACTAL_BLOCK_SIZE, FRACTAL_BLOCK_SIZE)
        rotary = _local_patch(rotary, grid_size, group_size, dim=0).flatten(0, 1)
        cos, sin = rotary[..., 0, 0], rotary[..., 1, 0]
        return cos.repeat_interleave(2, dim=-1).contiguous(), sin.repeat_interleave(2, dim=-1).contiguous()

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNKandinsky6SRTransformer3DModelConfig
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        return _Kandinsky6SRTimestepEmbedding(model).eval(), Kandinsky6SRTransformer3DModelWrapper(model).eval()

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNKandinsky6SRTransformer3DModelConfig):
        return {
            cfg.compiled_model_name: cls.compile(
                wrapped,
                rbln_compile_config=cfg,
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device_map[cfg.compiled_model_name],
            )
            for wrapped, cfg in zip(
                cls._wrap_model_if_needed(model, rbln_config), rbln_config.compile_cfgs, strict=True
            )
        }

    @classmethod
    def _create_runtimes(cls, compiled_models, rbln_config: RBLNKandinsky6SRTransformer3DModelConfig):
        return [
            rebel.Runtime(
                model,
                tensor_type="pt",
                device=rbln_config.device_map[cfg.compiled_model_name],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for model, cfg in zip(compiled_models, rbln_config.compile_cfgs, strict=True)
        ]

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        layout = pipe.get_tile_layout(rbln_config)
        if rbln_config.transformer.num_latent_frames is None:
            rbln_config.transformer.num_latent_frames = layout.num_latent_frames
        if rbln_config.transformer.sample_size is None:
            rbln_config.transformer.sample_size = layout.latent_base_tile_size
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor] | None,
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNKandinsky6SRTransformer3DModelConfig,
    ) -> RBLNKandinsky6SRTransformer3DModelConfig:
        missing = [key for key in ("num_latent_frames", "sample_size") if getattr(rbln_config, key) is None]
        if missing:
            raise ValueError(
                f"{', '.join(f'`{key}`' for key in missing)} must be specified to compile the Kandinsky 6 SR "
                "transformer. Compiling through a pipeline fills them from the input video; a standalone compile "
                "must pass them in rbln_config."
            )

        num_frames, height, width = cls._grid_size(model_config, rbln_config)
        if height % FRACTAL_BLOCK_SIZE or width % FRACTAL_BLOCK_SIZE:
            raise ValueError(
                f"NABLA attention needs the token grid {(height, width)} to be divisible by {FRACTAL_BLOCK_SIZE}."
            )
        seq_len = num_frames * height * width
        aligned_blocks = nabla_aligned_blocks(seq_len // NABLA_BLOCK_TOKENS)
        batch_size = rbln_config.batch_size
        in_channels = math.prod(model_config.patch_size) * (2 * model_config.in_visual_dim + 1)
        head_dim = sum(model_config.axes_dims)
        input_info = [
            ("hidden_states", [batch_size, seq_len, in_channels], rbln_config.dtype),
            ("modulation", [batch_size, model_config.num_visual_blocks, 6, model_config.model_dim], "float32"),
            ("out_modulation", [batch_size, 2, model_config.model_dim], "float32"),
            ("rope_cos", [seq_len, 1, head_dim], "float32"),
            ("rope_sin", [seq_len, 1, head_dim], "float32"),
            ("sta_mask", [aligned_blocks, aligned_blocks], "bool"),
        ]
        rbln_config.set_compile_cfgs(
            [
                RBLNCompileConfig(
                    compiled_model_name="timestep_embedding", input_info=[("timestep", [batch_size], "float32")]
                ),
                RBLNCompileConfig(input_info=input_info),
            ]
        )
        return rbln_config

    @property
    def compiled_batch_size(self):
        return self.rbln_config.compile_cfgs[-1].input_info[0][1][0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple:
        """
        Forward pass for the RBLN-optimized Kandinsky6SRTransformer3DModel.

        Args:
            hidden_states (torch.Tensor): Latent tiles in the (batch, frames, height, width, channels) layout: the
                noisy latent, the anchor latent and the anchor mask concatenated along the channel axis.
            timestep (torch.Tensor): Diffusion timestep on the [0, num_train_timesteps] scale, shaped (batch,).
            return_dict (bool): Whether to return a `Transformer2DModelOutput` instead of a plain tuple.

        Returns:
            (`~diffusers.models.modeling_outputs.Transformer2DModelOutput` | tuple)
        """
        batch_size = hidden_states.shape[0]
        if batch_size > self.compiled_batch_size:
            raise ValueError(
                f"The transformer is compiled for {self.compiled_batch_size} tiles per call, got {batch_size}. Call "
                "the pipeline with `tiles_batch_size` at most the compiled batch size."
            )
        patch_t, patch_h, patch_w = self.config.patch_size
        latent_frames, latent_height, latent_width, channels = hidden_states.shape[1:]
        grid_size = (latent_frames // patch_t, latent_height // patch_h, latent_width // patch_w)
        if grid_size != self.grid_size:
            raise ValueError(
                f"The transformer is compiled for a {self.grid_size} token grid of a "
                f"{(self.rbln_config.num_latent_frames, *self.rbln_config.sample_size)} latent tile, got a latent tile "
                f"of {(latent_frames, latent_height, latent_width)}. The video size, the frame count and "
                "`resolution_scale` of the call must match the compiled ones."
            )

        # The patchify of `Kandinsky6SRVisualEmbeddings` and the NABLA token order, on the host.
        tokens = hidden_states.view(
            batch_size, grid_size[0], patch_t, grid_size[1], patch_h, grid_size[2], patch_w, channels
        )
        tokens = tokens.permute(0, 1, 3, 5, 2, 4, 6, 7).flatten(4, 7)
        group_size = (1, FRACTAL_BLOCK_SIZE, FRACTAL_BLOCK_SIZE)
        tokens = _local_patch(tokens, grid_size, group_size, dim=1).flatten(1, 2)

        # The last chunk of the pipeline holds fewer tiles when the tile count is not a multiple of the batch.
        tokens = tokens.to(self.dtype)
        timestep = timestep.reshape(-1).expand(batch_size).float()
        padding = self.compiled_batch_size - batch_size
        if padding:
            tokens = torch.cat([tokens, tokens.new_zeros(padding, *tokens.shape[1:])])
            timestep = torch.cat([timestep, timestep.new_zeros(padding)])
        modulation, out_modulation = self.model[0](timestep.contiguous())
        output = self.model[1](
            tokens.contiguous(), modulation, out_modulation, self.rope_cos, self.rope_sin, self.sta_mask
        )[:batch_size]

        # The NABLA token order back to the grid and the unpatchify of `Kandinsky6SROutLayer`.
        output = _local_merge(
            output.reshape(batch_size, -1, NABLA_BLOCK_TOKENS, output.shape[-1]), grid_size, group_size, dim=1
        )
        output = (
            output.view(batch_size, *grid_size, -1, patch_t, patch_h, patch_w)
            .permute(0, 1, 5, 2, 6, 3, 7, 4)
            .flatten(1, 2)
            .flatten(2, 3)
            .flatten(3, 4)
        )
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


__all__ = [
    "RBLNKandinsky6SRTransformer3DModel",
]
