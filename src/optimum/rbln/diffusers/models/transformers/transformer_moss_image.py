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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from torch.nn import functional as F

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ...configurations import RBLNMOSSImageTransformerConfig
from ...pipelines.moss_image.pipeline_moss_image_upstream import (
    MOSSImageTransformer,
    _RoPE,
    _rotate,
    _situ,
    _validate_skip_layers,
)


if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


# The modules of the transformer that run on the host: the time embedding and the two modulations it feeds.
_HOST_MODULES = ("time_embedding", "shared_block_modulation", "video_head_modulation")


def _attention(q, k, v, rope, heads, attention_mask):
    # Upstream `_attention`, with the key mask of the padded text tokens.
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    hidden = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, attn_mask=attention_mask)
    return hidden.transpose(1, 2).flatten(2)


def _text_block(block, x, rope, attention_mask):
    # Upstream `_TextBlock.forward`, with the key mask.
    q, k, v = block.qkv(block.norm1(x)).chunk(3, -1)
    q = block.q_norm(q.unflatten(-1, (block.heads, -1))).flatten(2)
    k = block.k_norm(k.unflatten(-1, (block.heads, -1))).flatten(2)
    x = x + block.proj(_attention(q, k, v, rope, block.heads, attention_mask))
    y = block.norm2(x)
    return x + block.ffn_down(_situ(block.ffn_up(y), block.ffn_gate(y)))


def _block(block, x, rope, modulation, attention_mask):
    # Upstream `_Block.forward`, with the key mask.
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
    y = block.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
    dtype = block.proj["q"].weight.dtype
    y = y.to(dtype)
    q = block.q_norm(block.proj["q"](y).unflatten(-1, (block.heads, -1))).flatten(2)
    k = block.k_norm(block.proj["k"](y).unflatten(-1, (block.heads, -1))).flatten(2)
    x = x + block.proj["o"](_attention(q, k, block.proj["v"](y), rope, block.heads, attention_mask)) * gate_a[:, None]
    y = block.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
    hidden, gate = block.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
    return x + block.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]


class MOSSImageTransformerWrapper(torch.nn.Module):
    """
    The transformer over a fixed number of image and text tokens.

    Upstream runs each sample at its own prompt length. Here the prompt is padded to the compiled length and the
    padding is masked out of every attention, so the image tokens see the same keys. The time embedding, the
    modulations, the RoPE table and the mask come from the host (`RBLNMOSSImageTransformer.forward`). `layer_keep`
    holds one entry per block for skip-layer guidance: a block whose entry is 0 passes its input through.
    """

    def __init__(self, model: MOSSImageTransformer, num_image_tokens: int):
        super().__init__()
        self.model = model
        self.num_image_tokens = num_image_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        block_modulation: torch.Tensor,
        head_modulation: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_keep: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        image_tokens = self.num_image_tokens
        video = model.video_embedder(hidden_states)
        text = model.text_embedder.proj(encoder_hidden_states)
        for block in model.text_embedder.blocks:
            text = _text_block(block, text, rotary_emb[:, image_tokens:], attention_mask[..., image_tokens:])
        x = torch.cat((video, text), 1)
        for index, layer in enumerate(model.layers):
            keep = layer_keep[index]
            # Exact at 0 and at 1, so a skipped block passes `x` through unchanged.
            x = x * (1 - keep) + _block(layer, x, rotary_emb, block_modulation, attention_mask) * keep
        x = model.final_norm_video(x[:, :image_tokens].to(model.final_norm_video.weight.dtype))
        x = x * (1 + head_modulation[:, 1, None]) + head_modulation[:, 0, None]
        return model.final_linear_video(x.to(model.final_linear_video.weight.dtype))


class RBLNMOSSImageTransformer(RBLNModel):
    """
    RBLN implementation of the MOSS-Image transformer, `MOSSImageTransformer` of the checkpoint's remote code.

    The blocks run on the NPU for one latent size and one prompt length. The time embedding and its modulations, the
    RoPE table and the key mask of the padded prompt are computed on the host for each call, as is the skip-layer
    guidance: `skip_layers` becomes a per-block keep mask, so one compiled graph serves every skip set.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = MOSSImageTransformer
    _rbln_config_class = RBLNMOSSImageTransformerConfig
    _output_class = Transformer2DModelOutput

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        config = self.config
        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(config.freq_dim, config.dim), torch.nn.SiLU(), torch.nn.Linear(config.dim, config.dim)
        )
        self.shared_block_modulation = torch.nn.Linear(config.dim, 6 * config.dim)
        self.video_head_modulation = torch.nn.Linear(config.dim, 2 * config.dim)
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        for name in _HOST_MODULES:
            getattr(self, name).load_state_dict(artifacts[name], assign=True)
        self.rope = _RoPE(config.dim // config.num_heads, config.rope_base)

    @property
    def layers(self) -> range:
        # The pipeline checks skip-layer indices against `len(self.transformer.layers)`.
        return range(self.config.num_layers)

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: "Path",
        subfolder: str,
        rbln_config: RBLNMOSSImageTransformerConfig,
    ):
        save_dict = {name: getattr(model, name).state_dict() for name in _HOST_MODULES}
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        return rbln_config

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNMOSSImageTransformerConfig
    ) -> torch.nn.Module:
        height, width = rbln_config.sample_size
        return MOSSImageTransformerWrapper(model, height * width).eval()

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor] | None,
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNMOSSImageTransformerConfig,
    ) -> RBLNMOSSImageTransformerConfig:
        batch_size = rbln_config.batch_size
        height, width = rbln_config.sample_size
        num_tokens = height * width + rbln_config.max_sequence_length
        dim = model_config.dim
        input_info = [
            ("hidden_states", [batch_size, height * width, model_config.latent_channels], rbln_config.dtype),
            (
                "encoder_hidden_states",
                [batch_size, rbln_config.max_sequence_length, model_config.text_dim],
                rbln_config.dtype,
            ),
            ("rotary_emb", [batch_size, num_tokens, dim // model_config.num_heads], rbln_config.dtype),
            ("block_modulation", [batch_size, 6, dim], rbln_config.dtype),
            ("head_modulation", [batch_size, 2, dim], rbln_config.dtype),
            ("attention_mask", [batch_size, 1, 1, num_tokens], rbln_config.dtype),
            ("layer_keep", [model_config.num_layers], rbln_config.dtype),
        ]
        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def _modulation(self, timestep: torch.Tensor, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        # The time embedding as upstream computes it: a float64 sinusoid fed to the linears at their weight dtype.
        times = torch.as_tensor(timestep, dtype=torch.float32).reshape(-1).expand(batch)
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        block = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype))
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype))
        return block.view(batch, 6, -1), head.view(batch, 2, -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        skip_layers: Sequence[int] | None = None,
    ) -> Transformer2DModelOutput | tuple:
        """
        Forward pass for the RBLN-optimized MOSS-Image transformer.

        Args:
            hidden_states (torch.Tensor): The noisy latents, shaped (batch, latent_channels, height, width).
            timestep (torch.Tensor): The current denoising step.
            encoder_hidden_states (torch.Tensor): The text embeddings, shaped (batch, tokens, text_dim).
            encoder_attention_mask (torch.Tensor | None): A binary (batch, tokens) mask of the valid text tokens.
            return_dict (bool): Whether to return a `Transformer2DModelOutput` instead of a plain tuple.
            skip_layers (Sequence[int] | None): The blocks to skip, for skip-layer guidance.

        Returns:
            (`~diffusers.models.modeling_outputs.Transformer2DModelOutput` | tuple)
        """
        config = self.config
        skip_layers = _validate_skip_layers(skip_layers, config.num_layers)
        if hidden_states.ndim != 4 or hidden_states.shape[1] != config.latent_channels:
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
        if batch != self.rbln_config.batch_size:
            raise ValueError(
                f"The transformer was compiled for a batch of {self.rbln_config.batch_size} and got {batch}. The "
                "batch is the number of prompts times num_images_per_prompt."
            )
        if (height, width) != tuple(self.rbln_config.sample_size):
            raise ValueError(
                f"The transformer was compiled for a {self.rbln_config.sample_size} latent and got {(height, width)}."
            )

        dtype = self.rbln_config.dtype
        max_sequence_length = self.rbln_config.max_sequence_length
        image_tokens = height * width
        # Upstream evaluates each sample at its own prompt length. Here the valid tokens of each sample move to the
        # front of a prompt of the compiled length and the rest is masked, which gives the image tokens the same keys.
        text = encoder_hidden_states.new_zeros(batch, max_sequence_length, config.text_dim)
        valid = torch.zeros(batch, max_sequence_length, dtype=torch.bool)
        rotary_emb = torch.empty(batch, image_tokens + max_sequence_length, config.dim // config.num_heads)
        for index in range(batch):
            context = encoder_hidden_states[index][mask[index].bool()]
            length = context.shape[0]
            if length == 0:
                raise ValueError("Each sample needs at least one valid text token.")
            if length > max_sequence_length:
                raise ValueError(
                    f"The prompt has {length} tokens and the transformer was compiled for {max_sequence_length}. "
                    "Compile with a larger max_sequence_length."
                )
            text[index, :length] = context
            valid[index, :length] = True
            # The image tokens take the true prompt length as their frame coordinate, the text tokens their position.
            rope = self.rope(height, width, length, hidden_states.device)
            rotary_emb[index, :image_tokens] = rope[0, :image_tokens]
            rope = self.rope(height, width, max_sequence_length, hidden_states.device)
            rotary_emb[index, image_tokens:] = rope[0, image_tokens:]
        attention_mask = torch.zeros(batch, 1, 1, image_tokens + max_sequence_length, dtype=dtype)
        attention_mask[:, 0, 0, image_tokens:].masked_fill_(~valid, torch.finfo(dtype).min)
        layer_keep = torch.ones(config.num_layers, dtype=dtype)
        layer_keep[list(skip_layers)] = 0
        block_modulation, head_modulation = self._modulation(timestep, batch)

        sample = self.model[0](
            hidden_states.flatten(2).transpose(1, 2).to(dtype).contiguous(),
            text.to(dtype).contiguous(),
            rotary_emb.to(dtype).contiguous(),
            block_modulation.to(dtype).contiguous(),
            head_modulation.to(dtype).contiguous(),
            attention_mask.contiguous(),
            layer_keep.contiguous(),
        )
        if isinstance(sample, (tuple, list)):
            sample = sample[0]
        sample = sample.transpose(1, 2).reshape(batch, -1, height, width)
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)


__all__ = [
    "RBLNMOSSImageTransformer",
]
