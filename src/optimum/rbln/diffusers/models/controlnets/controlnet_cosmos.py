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

from pathlib import Path
from typing import TYPE_CHECKING, Union

import torch
from diffusers.models.controlnets.controlnet_cosmos import CosmosControlNetModel, CosmosControlNetOutput
from diffusers.models.transformers.transformer_cosmos import CosmosEmbedding, CosmosPatchEmbed
from torchvision import transforms

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ...configurations import RBLNCosmosControlNetModelConfig
from ..transformers.transformer_cosmos import RBLNCosmosRotaryPosEmbed, RBLNTimesteps


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


class CosmosControlNetWrapper(torch.nn.Module):
    """Compile graph: the ControlNet block stack. Embeddings are computed on the host."""

    def __init__(self, model: CosmosControlNetModel):
        super().__init__()
        self.control_blocks = model.control_blocks
        self.uses_img_context = model.config.img_context_dim_in is not None and model.config.img_context_dim_in > 0

    def forward(
        self,
        control_hidden_states: torch.Tensor,
        base_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb_0: torch.Tensor,
        image_rotary_emb_1: torch.Tensor,
        conditioning_scales: torch.Tensor,
        img_context: torch.Tensor | None = None,
    ):
        image_rotary_emb = [image_rotary_emb_0, image_rotary_emb_1]
        # The cross-attention processor of Transfer2.5 blocks expects a (text, img) tuple.
        context = (encoder_hidden_states, img_context) if self.uses_img_context else encoder_hidden_states

        outputs = []
        hidden_states = control_hidden_states
        for block_idx, block in enumerate(self.control_blocks):
            hidden_states, control_proj = block(
                hidden_states=hidden_states,
                encoder_hidden_states=context,
                embedded_timestep=embedded_timestep,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                extra_pos_emb=None,
                attention_mask=None,
                controlnet_residual=None,
                latents=base_hidden_states,
                block_idx=block_idx,
            )
            outputs.append(control_proj * conditioning_scales[block_idx : block_idx + 1].reshape(1, 1, 1))
        return tuple(outputs)


class RBLNCosmosControlNetModel(RBLNModel):
    """
    RBLN implementation of CosmosControlNetModel (Cosmos-Transfer2.5).

    The block stack runs on the NPU; patch/time/rope/projection embeddings run on the host,
    mirroring RBLNCosmosTransformer3DModel. Its forward keeps the upstream signature and
    returns the per-block control residuals that the transformer injects every
    `controlnet_block_every_n` blocks.
    """

    hf_library_name = "diffusers"
    auto_model_class = CosmosControlNetModel
    _rbln_config_class = RBLNCosmosControlNetModelConfig

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)

        hidden_size = self.config.model_channels
        self.rope = RBLNCosmosRotaryPosEmbed(
            hidden_size=self.config.attention_head_dim,
            max_size=self.config.max_size,
            patch_size=self.config.patch_size,
            rope_scale=self.config.rope_scale,
        )
        self.rope.load_state_dict(artifacts["rope"])

        self.patch_embed = CosmosPatchEmbed(self.config.in_channels, hidden_size, self.config.patch_size, bias=False)
        self.patch_embed.load_state_dict(artifacts["patch_embed"])
        self.patch_embed.to(self.rbln_config.dtype)

        self.patch_embed_base = CosmosPatchEmbed(
            self.config.latent_channels, hidden_size, self.config.patch_size, bias=False
        )
        self.patch_embed_base.load_state_dict(artifacts["patch_embed_base"])
        self.patch_embed_base.to(self.rbln_config.dtype)

        self.time_embed = CosmosEmbedding(hidden_size, hidden_size)
        time_proj = self.time_embed.time_proj
        self.time_embed.time_proj = RBLNTimesteps(
            time_proj.num_channels, time_proj.flip_sin_to_cos, time_proj.downscale_freq_shift, time_proj.scale
        )
        self.time_embed.load_state_dict(artifacts["time_embed"])
        self.time_embed.to(self.rbln_config.dtype)

        if artifacts.get("crossattn_proj") is None:
            self.crossattn_proj = None
        else:
            self.crossattn_proj = torch.nn.Sequential(
                torch.nn.Linear(
                    self.config.crossattn_proj_in_channels, self.config.encoder_hidden_states_channels, bias=True
                ),
                torch.nn.GELU(),
            )
            self.crossattn_proj.load_state_dict(artifacts["crossattn_proj"])
            self.crossattn_proj.to(self.rbln_config.dtype)

        if artifacts.get("img_context_proj") is None:
            self.img_context_proj = None
        else:
            self.img_context_proj = torch.nn.Sequential(
                torch.nn.Linear(self.config.img_context_dim_in, self.config.img_context_dim_out, bias=True),
                torch.nn.GELU(),
            )
            self.img_context_proj.load_state_dict(artifacts["img_context_proj"])
            self.img_context_proj.to(self.rbln_config.dtype)

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        save_dict = {}
        save_dict["rope"] = model.rope.state_dict()
        save_dict["patch_embed"] = model.patch_embed.state_dict()
        save_dict["patch_embed_base"] = model.patch_embed_base.state_dict()
        save_dict["time_embed"] = model.time_embed.state_dict()
        if model.crossattn_proj is not None:
            save_dict["crossattn_proj"] = model.crossattn_proj.state_dict()
        if model.img_context_proj is not None:
            save_dict["img_context_proj"] = model.img_context_proj.state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        return CosmosControlNetWrapper(model=model).eval()

    def compute_embedding(
        self,
        controls_latents: torch.Tensor,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        condition_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        img_context: torch.Tensor | None = None,
        fps: int | None = None,
    ):
        # Mirrors the host-side part of CosmosControlNetModel.forward (steps 1-8).
        B, _, T, H, W = controls_latents.shape
        dtype = self.rbln_config.dtype
        controls_latents = controls_latents.to(dtype)
        latents = latents.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        # 1. Control latents: zero-pad the modality slots, then append cond/padding masks.
        control_hidden_states = controls_latents
        vace_in_channels = self.config.in_channels - 1
        if control_hidden_states.shape[1] < vace_in_channels - 1:
            pad_c = vace_in_channels - 1 - control_hidden_states.shape[1]
            control_hidden_states = torch.cat(
                [control_hidden_states, torch.zeros((B, pad_c, T, H, W), dtype=dtype)], dim=1
            )
        if condition_mask is not None:
            control_hidden_states = torch.cat([control_hidden_states, condition_mask.to(dtype)], dim=1)
        else:
            control_hidden_states = torch.cat(
                [control_hidden_states, torch.zeros_like(controls_latents[:, :1])], dim=1
            )
        padding_mask_resized = transforms.functional.resize(
            padding_mask, list(control_hidden_states.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
        ).to(dtype)
        control_hidden_states = torch.cat(
            [control_hidden_states, padding_mask_resized.unsqueeze(2).repeat(B, 1, T, 1, 1)], dim=1
        )

        # 2. Base latents: latents + cond mask + padding mask.
        base_hidden_states = latents
        if condition_mask is not None:
            base_hidden_states = torch.cat([base_hidden_states, condition_mask.to(dtype)], dim=1)
        base_hidden_states = torch.cat(
            [base_hidden_states, padding_mask_resized.unsqueeze(2).repeat(B, 1, T, 1, 1)], dim=1
        )

        # 3. RoPE (shape-dependent only).
        image_rotary_emb = self.rope(control_hidden_states, fps=fps)

        # 4. Patchify both streams.
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = T // p_t
        post_patch_height = H // p_h
        post_patch_width = W // p_w
        control_hidden_states = self.patch_embed(control_hidden_states).flatten(1, 3)
        base_hidden_states = self.patch_embed_base(base_hidden_states).flatten(1, 3)

        # 5. Timestep embeddings (Transfer2.5 feeds per-frame timesteps, [B, 1, T, 1, 1]).
        if timestep.ndim == 1:
            temb, embedded_timestep = self.time_embed(base_hidden_states, timestep)
        elif timestep.ndim == 5:
            assert timestep.shape == (B, 1, T, 1, 1), (
                f"Expected timestep to have shape [B, 1, T, 1, 1], but got {timestep.shape}"
            )
            temb, embedded_timestep = self.time_embed(base_hidden_states, timestep.flatten())
            temb, embedded_timestep = (
                x.view(B, post_patch_num_frames, 1, 1, -1)
                .expand(-1, -1, post_patch_height, post_patch_width, -1)
                .flatten(1, 3)
                for x in (temb, embedded_timestep)
            )
        else:
            raise AssertionError("Unsupported shape of `timestep`")

        # 6. Context projections.
        if self.crossattn_proj is not None:
            encoder_hidden_states = self.crossattn_proj(encoder_hidden_states)
        if self.img_context_proj is not None:
            if img_context is None:
                img_context = torch.zeros(
                    (B, self.rbln_config.img_context_num_tokens, self.config.img_context_dim_in), dtype=dtype
                )
            img_context = self.img_context_proj(img_context.to(dtype))

        return (
            control_hidden_states,
            base_hidden_states,
            encoder_hidden_states,
            embedded_timestep,
            temb,
            image_rotary_emb[0],
            image_rotary_emb[1],
            img_context,
        )

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        # The ControlNet shares its geometry with the transformer submodule.
        controlnet = getattr(rbln_config, submodule_name)
        transformer = rbln_config.transformer
        for name in ("num_frames", "height", "width", "num_latent_frames", "latent_height", "latent_width"):
            if getattr(controlnet, name) is None:
                setattr(controlnet, name, getattr(transformer, name))
        controlnet.max_seq_len = transformer.max_seq_len
        controlnet.embedding_dim = transformer.embedding_dim
        if controlnet.img_context_num_tokens is None:
            controlnet.img_context_num_tokens = getattr(pipe.transformer.config, "img_context_num_tokens", None)
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNCosmosControlNetModelConfig,
    ) -> RBLNCosmosControlNetModelConfig:
        missing = [
            name
            for name in ("num_latent_frames", "latent_height", "latent_width", "max_seq_len", "embedding_dim")
            if getattr(rbln_config, name) is None
        ]
        if missing:
            raise ValueError(f"{', '.join(missing)} must be specified to compile RBLNCosmosControlNetModel.")

        p_t, p_h, p_w = model_config.patch_size
        hidden_dim = (
            (rbln_config.num_latent_frames // p_t)
            * (rbln_config.latent_height // p_h)
            * (rbln_config.latent_width // p_w)
        )
        hidden_size = model_config.model_channels
        n_blocks = model_config.n_controlnet_blocks

        input_info = [
            ("control_hidden_states", [rbln_config.batch_size, hidden_dim, hidden_size], rbln_config.dtype),
            ("base_hidden_states", [rbln_config.batch_size, hidden_dim, hidden_size], rbln_config.dtype),
            (
                "encoder_hidden_states",
                [rbln_config.batch_size, rbln_config.max_seq_len, rbln_config.embedding_dim],
                rbln_config.dtype,
            ),
            # Transfer2.5 always feeds per-frame timesteps ([B, 1, T, 1, 1]).
            ("embedded_timestep", [rbln_config.batch_size, hidden_dim, hidden_size], rbln_config.dtype),
            ("temb", [1, hidden_dim, hidden_size * 3], rbln_config.dtype),
            ("image_rotary_emb_0", [hidden_dim, model_config.attention_head_dim], "float32"),
            ("image_rotary_emb_1", [hidden_dim, model_config.attention_head_dim], "float32"),
            ("conditioning_scales", [n_blocks], rbln_config.dtype),
        ]
        if model_config.img_context_dim_in is not None and model_config.img_context_dim_in > 0:
            if rbln_config.img_context_num_tokens is None:
                # The token count lives on the transformer config (the pipeline hook copies it);
                # 256 is the CosmosTransformer3DModel default.
                rbln_config.img_context_num_tokens = 256
            input_info.append(
                (
                    "img_context",
                    [rbln_config.batch_size, rbln_config.img_context_num_tokens, model_config.img_context_dim_out],
                    rbln_config.dtype,
                )
            )

        compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([compile_config])
        return rbln_config

    def forward(
        self,
        controls_latents: torch.Tensor,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | tuple,
        condition_mask: torch.Tensor,
        conditioning_scale: float | list[float] = 1.0,
        padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        fps: int | None = None,
        return_dict: bool = True,
    ):
        if isinstance(encoder_hidden_states, tuple):
            encoder_hidden_states, img_context = encoder_hidden_states
        else:
            img_context = None

        n_blocks = self.config.n_controlnet_blocks
        if not isinstance(conditioning_scale, list):
            scales = [conditioning_scale] * n_blocks
        else:
            scales = (conditioning_scale * n_blocks)[:n_blocks]
        conditioning_scales = torch.tensor(scales, dtype=self.rbln_config.dtype)

        (
            control_hidden_states,
            base_hidden_states,
            encoder_hidden_states,
            embedded_timestep,
            temb,
            rope_0,
            rope_1,
            img_context,
        ) = self.compute_embedding(
            controls_latents, latents, timestep, encoder_hidden_states, condition_mask, padding_mask, img_context, fps
        )

        inputs = [
            control_hidden_states,
            base_hidden_states,
            encoder_hidden_states,
            embedded_timestep,
            temb,
            rope_0,
            rope_1,
            conditioning_scales,
        ]
        if img_context is not None:
            inputs.append(img_context)

        result = list(self.model[0](*inputs))

        if not return_dict:
            return (result,)
        return CosmosControlNetOutput(control_block_samples=result)
