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

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import rebel
import torch
from diffusers import CosmosTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_cosmos import (
    CosmosEmbedding,
    CosmosLearnablePositionalEmbed,
    CosmosPatchEmbed,
    CosmosRotaryPosEmbed,
)
from torchvision import transforms

from ....configuration_utils import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...configurations import RBLNCosmosTransformer3DModelConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNCosmosTransformer3DModelConfig, RBLNDiffusionMixin, RBLNDiffusionMixinConfig


logger = get_logger(__name__)


class CosmosTransformer3DModelWrapper(torch.nn.Module):
    def __init__(
        self,
        model: CosmosTransformer3DModel,
        num_latent_frames: int = 16,
        latent_height: int = 88,
        latent_width: int = 160,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_latent_frames = num_latent_frames
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.p_t, self.p_h, self.p_w = model.config.patch_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb_0: torch.Tensor,
        image_rotary_emb_1: torch.Tensor,
        extra_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        image_rotary_emb = [image_rotary_emb_0, image_rotary_emb_1]
        for block in self.model.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                embedded_timestep=embedded_timestep,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                extra_pos_emb=extra_pos_emb,
                attention_mask=attention_mask,
            )
        post_patch_num_frames = self.num_latent_frames // self.p_t
        post_patch_height = self.latent_height // self.p_h
        post_patch_width = self.latent_width // self.p_w
        hidden_states = self.model.norm_out(hidden_states, embedded_timestep, temb)
        hidden_states = self.model.proj_out(hidden_states)
        hidden_states = hidden_states.unflatten(2, (self.p_h, self.p_w, self.p_t, -1))
        hidden_states = hidden_states.unflatten(1, (post_patch_num_frames, post_patch_height, post_patch_width))
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return (hidden_states,)


class RBLNCosmosTransformer3DModel(RBLNModel):
    """
    RBLN implementation of CosmosTransformer3DModel for diffusion models like Cosmos.

    The CosmosTransformer3DModel takes text and/or image embeddings from encoders (like CLIP) and
    maps them to a shared latent space that guides the diffusion process to generate the desired image.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = CosmosTransformer3DModel

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)

        hidden_size = self.config.num_attention_heads * self.config.attention_head_dim
        patch_embed_in_channels = (
            self.config.in_channels + 1 if self.config.concat_padding_mask else self.config.in_channels
        )
        self.rope = CosmosRotaryPosEmbed(
            hidden_size=self.config.attention_head_dim,
            max_size=self.config.max_size,
            patch_size=self.config.patch_size,
            rope_scale=self.config.rope_scale,
        )
        self.rope.load_state_dict(artifacts["rope"])
        if artifacts["learnable_pos_embed"] is None:
            self.learnable_pos_embed = None
        else:
            self.learnable_pos_embed = CosmosLearnablePositionalEmbed(
                hidden_size=hidden_size,
                max_size=self.config.max_size,
                patch_size=self.config.patch_size,
            )
            self.learnable_pos_embed.load_state_dict(artifacts["learnable_pos_embed"])
        self.patch_embed = CosmosPatchEmbed(patch_embed_in_channels, hidden_size, self.config.patch_size, bias=False)
        self.patch_embed.load_state_dict(artifacts["patch_embed"])
        self.time_embed = CosmosEmbedding(hidden_size, hidden_size)
        self.time_embed.load_state_dict(artifacts["time_embed"])

    def compute_embedding(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fps: Optional[int] = None,
        condition_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # 1. Concatenate padding mask if needed & prepare attention mask
        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        if self.config.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(hidden_states.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            hidden_states = torch.cat(
                [hidden_states, padding_mask.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1)], dim=1
            )

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

        # 2. Generate positional embeddings
        image_rotary_emb = self.rope(hidden_states, fps=fps)
        extra_pos_emb = self.learnable_pos_embed(hidden_states) if self.config.extra_pos_embed_type else None

        # 3. Patchify input
        p_t, p_h, p_w = self.config.patch_size
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.flatten(1, 3)  # [B, T, H, W, C] -> [B, THW, C]

        # 4. Timestep embeddings
        temb, embedded_timestep = self.time_embed(hidden_states, timestep)

        return (
            hidden_states,
            temb,
            embedded_timestep,
            image_rotary_emb[0],
            image_rotary_emb[1],
            extra_pos_emb,
            attention_mask,
        )

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        num_latent_frames = rbln_config.num_latent_frames
        latent_height = rbln_config.latent_height
        latent_width = rbln_config.latent_width
        return CosmosTransformer3DModelWrapper(
            model=model,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
        ).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> RBLNCosmosTransformer3DModelConfig:
        rbln_config.transformer.num_latent_frames = (
            rbln_config.transformer.num_frames - 1
        ) // pipe.vae_scale_factor_temporal + 1
        rbln_config.transformer.latent_height = rbln_config.transformer.height // pipe.vae_scale_factor_spatial
        rbln_config.transformer.latent_width = rbln_config.transformer.width // pipe.vae_scale_factor_spatial
        rbln_config.transformer.max_seq_len = pipe.text_encoder.config.n_positions
        rbln_config.transformer.embedding_dim = pipe.text_encoder.encoder.embed_tokens.embedding_dim

        return rbln_config

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
        if model.learnable_pos_embed is not None:
            save_dict["learnable_pos_embed"] = model.learnable_pos_embed.state_dict()
        save_dict["patch_embed"] = model.patch_embed.state_dict()
        save_dict["time_embed"] = model.time_embed.state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: "RBLNCosmosTransformer3DModelConfig",
    ) -> RBLNCosmosTransformer3DModelConfig:
        p_t, p_h, p_w = model_config.patch_size
        hidden_dim = (
            (rbln_config.num_latent_frames // p_t)
            * (rbln_config.latent_height // p_h)
            * (rbln_config.latent_width // p_w)
        )
        attention_head_dim = model_config.attention_head_dim
        hidden_size = model.config.num_attention_heads * model.config.attention_head_dim
        input_info = [
            (
                "hidden_states",
                [
                    rbln_config.batch_size,
                    hidden_dim,
                    hidden_size,
                ],
                "float32",
            ),
            (
                "encoder_hidden_states",
                [
                    rbln_config.batch_size,
                    rbln_config.max_seq_len,
                    rbln_config.embedding_dim,
                ],
                "float32",
            ),
            ("embedded_timestep", [rbln_config.batch_size, hidden_size], "float32"),
            ("temb", [1, hidden_size * 3], "float32"),
            ("image_rotary_emb_0", [hidden_dim, attention_head_dim], "float32"),
            ("image_rotary_emb_1", [hidden_dim, attention_head_dim], "float32"),
            ("extra_pos_emb", [rbln_config.batch_size, hidden_dim, hidden_size], "float32"),
        ]

        compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([compile_config])
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNModelConfig,
    ) -> List[rebel.Runtime]:
        if DEFAULT_COMPILED_MODEL_NAME not in rbln_config.device_map:
            cls._raise_missing_compiled_file_error([DEFAULT_COMPILED_MODEL_NAME])

        return [
            rebel.Runtime(
                compiled_model,
                tensor_type="pt",
                device=rbln_config.device_map[DEFAULT_COMPILED_MODEL_NAME],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for compiled_model in compiled_models
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fps: Optional[int] = None,
        condition_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Forward pass for the RBLN-optimized CosmosTransformer3DModel.

        Args:
            hidden_states (torch.Tensor): The currently predicted image embeddings.
            timestep (torch.Tensor): Current denoising step.
            encoder_hidden_states (torch.Tensor): Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            fps: (Optional[int]): Frames per second for the video being generated.
            condition_mask (Optional[torch.Tensor]): Tensor of condition mask.
            padding_mask (Optional[torch.Tensor]): Tensor of padding mask.
            return_dict (bool): Whether or not to return a [`~diffusers.models.modeling_output.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            (Union[`~diffusers.models.modeling_output.Transformer2DModelOutput`, Tuple])
        """
        (
            hidden_states,
            temb,
            embedded_timestep,
            image_rotary_emb_0,
            image_rotary_emb_1,
            extra_pos_emb,
            attention_mask,
        ) = self.compute_embedding(hidden_states, timestep, attention_mask, fps, condition_mask, padding_mask)

        hidden_states = self.model[0].forward(
            hidden_states,
            encoder_hidden_states,
            embedded_timestep,
            temb,
            image_rotary_emb_0,
            image_rotary_emb_1,
            extra_pos_emb,
        )

        if not return_dict:
            return (hidden_states,)
        else:
            return Transformer2DModelOutput(sample=hidden_states)
