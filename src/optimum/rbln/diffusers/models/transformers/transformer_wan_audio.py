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
from typing import TYPE_CHECKING, Any

import torch

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ...configurations import RBLNWanAudioModelConfig
from ...pipelines.moss_soundeffect.pipeline_moss_soundeffect_upstream import (
    WanAudioModel,
    attention,
    modulate,
    precompute_freqs_cis_1d,
    sinusoidal_embedding_1d,
)


if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


class WanAudioModelWrapper(torch.nn.Module):
    def __init__(self, model: WanAudioModel):
        super().__init__()
        self.model = model
        for block in model.blocks:
            self._permute_head_channels(block.self_attn)

    @staticmethod
    def _permute_head_channels(attention: torch.nn.Module) -> None:
        # Upstream `rope_apply`'s x[..., 0] / x[..., 1] selects fall to the host; (evens, odds) allows rotate-half.
        num_heads, head_dim = attention.num_heads, attention.head_dim
        within_head = torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])
        index = (torch.arange(num_heads)[:, None] * head_dim + within_head).reshape(-1)
        with torch.no_grad():
            for module in (attention.q, attention.k, attention.norm_q, attention.norm_k):
                module.weight.copy_(module.weight[index])
                if getattr(module, "bias", None) is not None:
                    module.bias.copy_(module.bias[index])

    @staticmethod
    def _rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, num_heads: int) -> torch.Tensor:
        out_dtype = x.dtype
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, num_heads, -1).float()
        x_even, x_odd = x.chunk(2, dim=-1)
        out = torch.cat((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
        return out.flatten(2).to(out_dtype)

    def _block_forward(self, block, x, context, vectors, cos, sin):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = vectors
        attn = block.self_attn
        hidden = modulate(block.norm1(x), shift_msa, scale_msa)
        q = self._rope_apply(attn.norm_q(attn.q(hidden)), cos, sin, attn.num_heads)
        k = self._rope_apply(attn.norm_k(attn.k(hidden)), cos, sin, attn.num_heads)
        x = x + gate_msa * attn.o(attention(q, k, attn.v(hidden), attn.num_heads))
        x = x + block.cross_attn(block.norm3(x), context)
        hidden = modulate(block.norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp * block.ffn(hidden)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        modulation: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        context = model.text_embedding(context)
        x, (frames,) = model.patchify(x)
        # One [B, 6 * num_layers + 2, dim] input: a per-block [B, 6, dim] add + chunk in the graph falls to the host.
        vectors = [vector.unsqueeze(1) for vector in modulation.unbind(1)]
        for i, block in enumerate(model.blocks):
            x = self._block_forward(block, x, context, vectors[6 * i : 6 * (i + 1)], cos, sin)
        shift, scale = vectors[-2:]
        x = model.head.head(model.head.norm(x) * (1 + scale) + shift)
        return model.unpatchify(x, (frames,))


class RBLNWanAudioModel(RBLNModel):
    """
    RBLN implementation of WanAudioModel for text-to-audio models like MOSS-SoundEffect.

    The WanAudioModel denoises audio latents with self-attention over the latent frames and cross-attention to the
    text embeddings. The time embedding, its modulation and the RoPE table are computed on the host.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = WanAudioModel
    # The base class looks the HF class up by name in `diffusers`, which has no vendored class.
    _hf_class = WanAudioModel
    _rbln_config_class = RBLNWanAudioModelConfig

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)

        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(self.config.freq_dim, self.config.dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.config.dim, self.config.dim),
        )
        self.time_embedding.load_state_dict(artifacts["time_embedding"])
        self.time_embedding.to(self.rbln_config.dtype)
        self.time_projection = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(self.config.dim, self.config.dim * 6)
        )
        self.time_projection.load_state_dict(artifacts["time_projection"])
        self.time_projection.to(self.rbln_config.dtype)
        self.block_modulation = artifacts["block_modulation"].to(self.rbln_config.dtype)
        self.head_modulation = artifacts["head_modulation"].to(self.rbln_config.dtype)

    def compute_embedding(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ):
        batch_size, _, num_frames = x.shape
        timestep = torch.as_tensor(timestep).reshape(-1).expand(batch_size)
        t = self.time_embedding(sinusoidal_embedding_1d(self.config.freq_dim, timestep).to(self.rbln_config.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.config.dim))
        block_vectors = (self.block_modulation + t_mod).transpose(0, 1).reshape(batch_size, -1, self.config.dim)
        head_vectors = self.head_modulation + t.unsqueeze(1)
        modulation = torch.cat([block_vectors, head_vectors], dim=1)
        f = num_frames
        freqs = precompute_freqs_cis_1d(self.config.dim // self.config.num_heads)
        freqs = torch.cat(
            [
                freqs[0][:f].view(f, -1).expand(f, -1),
                freqs[1][:f].view(f, -1).expand(f, -1),
                freqs[2][:f].view(f, -1).expand(f, -1),
            ],
            dim=-1,
        ).reshape(f, 1, -1)
        cos, sin = freqs.real.float(), freqs.imag.float()

        dtype = self.rbln_config.dtype
        return x.to(dtype), context.to(dtype), modulation.to(dtype), cos, sin

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNWanAudioModelConfig) -> torch.nn.Module:
        return WanAudioModelWrapper(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        if rbln_config.transformer.num_latent_frames is None:
            seconds = rbln_config.transformer.max_inference_seconds or pipe.config.max_inference_seconds
            rbln_config.transformer.num_latent_frames = pipe.config.sample_rate * seconds // pipe.vae.hop_length
        return rbln_config

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: "Path",
        subfolder: str,
        rbln_config: RBLNWanAudioModelConfig,
    ):
        save_dict = {}
        save_dict["time_embedding"] = model.time_embedding.state_dict()
        save_dict["time_projection"] = model.time_projection.state_dict()
        save_dict["block_modulation"] = torch.stack([block.modulation.detach() for block in model.blocks])
        save_dict["head_modulation"] = model.head.modulation.detach()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNWanAudioModelConfig,
    ) -> RBLNWanAudioModelConfig:
        if rbln_config.num_latent_frames is None:
            raise ValueError("num_latent_frames must be specified to compile RBLNWanAudioModel.")
        num_frames = rbln_config.num_latent_frames
        rope_dim = model_config.dim // model_config.num_heads // 2
        input_info = [
            ("x", [rbln_config.batch_size, model_config.in_dim, num_frames], rbln_config.dtype),
            ("context", [rbln_config.batch_size, rbln_config.text_len, model_config.text_dim], rbln_config.dtype),
            (
                "modulation",
                [rbln_config.batch_size, 6 * model_config.num_layers + 2, model_config.dim],
                rbln_config.dtype,
            ),
            ("cos", [num_frames, 1, rope_dim], "float32"),
            ("sin", [num_frames, 1, rope_dim], "float32"),
        ]

        compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([compile_config])
        return rbln_config

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass for the RBLN-optimized WanAudioModel.

        Args:
            x (torch.Tensor): The noisy latents, shaped (batch, in_dim, frames).
            timestep (torch.Tensor): Current denoising step, one value or one per clip.
            context (torch.Tensor): The text embeddings, shaped (batch, text_len, text_dim).
            kwargs: Ignored.

        Returns:
            (torch.Tensor): The predicted velocity, shaped like `x`.
        """
        x, context, modulation, cos, sin = self.compute_embedding(x, timestep, context)
        return self.model[0](
            x.contiguous(), context.contiguous(), modulation.contiguous(), cos.contiguous(), sin.contiguous()
        )


__all__ = [
    "RBLNWanAudioModel",
]
