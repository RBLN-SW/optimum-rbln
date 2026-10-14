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
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ....utils.logging import get_logger
from ...configurations import RBLNCogVideoXTransformer3DModelConfig


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

logger = get_logger(__name__)


class CogVideoXTransformer3DModelWrapper(torch.nn.Module):
    def __init__(self, model: CogVideoXTransformer3DModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        image_rotary_emb_cos: torch.Tensor | None = None,
        image_rotary_emb_sin: torch.Tensor | None = None,
    ):
        image_rotary_emb = None
        if image_rotary_emb_cos is not None:
            image_rotary_emb = (image_rotary_emb_cos, image_rotary_emb_sin)
        return self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )


class RBLNCogVideoXTransformer3DModel(RBLNModel):
    """
    RBLN implementation of CogVideoXTransformer3DModel for text-to-video and image-to-video diffusion.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = CogVideoXTransformer3DModel
    _rbln_config_class = RBLNCogVideoXTransformer3DModelConfig
    _output_class = Transformer2DModelOutput

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNCogVideoXTransformer3DModelConfig
    ) -> torch.nn.Module:
        if getattr(model, "ofs_embedding", None) is not None or model.config.patch_size_t is not None:
            raise NotImplementedError(
                "RBLNCogVideoXTransformer3DModel supports CogVideoX 1.0 models (CogVideoX-2b, CogVideoX-5b-I2V); "
                "the ofs embedding and temporal patches of CogVideoX 1.5 are not supported."
            )
        return CogVideoXTransformer3DModelWrapper(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        transformer_config = rbln_config.transformer
        if transformer_config.num_frames is None:
            transformer_config.num_frames = rbln_config.vae.num_frames
        if transformer_config.sample_size is None and rbln_config.vae.height is not None:
            transformer_config.sample_size = (
                rbln_config.vae.height // pipe.vae_scale_factor_spatial,
                rbln_config.vae.width // pipe.vae_scale_factor_spatial,
            )
        transformer_config.vae_scale_factor_temporal = pipe.vae_scale_factor_temporal
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor] | None,
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNCogVideoXTransformer3DModelConfig,
    ) -> RBLNCogVideoXTransformer3DModelConfig:
        if rbln_config.num_frames is None:
            rbln_config.num_frames = model_config.sample_frames
        if rbln_config.sample_size is None:
            rbln_config.sample_size = (model_config.sample_height, model_config.sample_width)
        if rbln_config.max_text_seq_length is None:
            rbln_config.max_text_seq_length = model_config.max_text_seq_length
        if rbln_config.vae_scale_factor_temporal is None:
            rbln_config.vae_scale_factor_temporal = model_config.temporal_compression_ratio

        batch_size = rbln_config.batch_size
        latent_h, latent_w = rbln_config.sample_size
        input_info = [
            (
                "hidden_states",
                [batch_size, rbln_config.num_latent_frames, model_config.in_channels, latent_h, latent_w],
                rbln_config.dtype,
            ),
            (
                "encoder_hidden_states",
                [batch_size, rbln_config.max_text_seq_length, model_config.text_embed_dim],
                rbln_config.dtype,
            ),
            ("timestep", [batch_size], "float32"),
        ]
        if model_config.use_rotary_positional_embeddings:
            # The pipeline builds the 3D RoPE tables on the host (_prepare_rotary_positional_embeddings).
            p = model_config.patch_size
            rope_shape = [
                rbln_config.num_latent_frames * (latent_h // p) * (latent_w // p),
                model_config.attention_head_dim,
            ]
            input_info.append(("image_rotary_emb_cos", rope_shape, "float32"))
            input_info.append(("image_rotary_emb_sin", rope_shape, "float32"))
        compile_cfgs = [RBLNCompileConfig(input_info=input_info)]
        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @contextmanager
    def cache_context(self, name: str):
        # CogVideoXPipeline wraps each transformer call in diffusers' CacheMixin.cache_context.
        # The compiled graphs carry no diffusers cache hooks, so there is nothing to set.
        yield

    @property
    def compiled_batch_size(self):
        return self.rbln_config.compile_cfgs[0].input_info[0][1][0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: int | float | torch.LongTensor,
        timestep_cond: torch.Tensor | None = None,
        ofs: int | float | torch.LongTensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple:
        """
        Forward pass for the RBLN-optimized CogVideoXTransformer3DModel.

        Args:
            hidden_states (torch.Tensor): The noisy latents, shaped (batch, frames, channels, height, width).
            encoder_hidden_states (torch.Tensor): The text embeddings.
            timestep (int | float | torch.LongTensor): The current denoising step.
            timestep_cond (torch.Tensor | None): Not supported; must be None.
            ofs (int | float | torch.LongTensor | None): Not supported; must be None.
            image_rotary_emb (tuple[torch.Tensor, torch.Tensor] | None): The RoPE (cos, sin) tables. Required by models
                with rotary positional embeddings (e.g. CogVideoX-5b-I2V); must be None otherwise.
            attention_kwargs (dict[str, Any] | None): Ignored.
            return_dict (bool): Whether to return a `Transformer2DModelOutput` instead of a plain tuple.

        Returns:
            (`~diffusers.models.modeling_outputs.Transformer2DModelOutput` | tuple)
        """
        if timestep_cond is not None or ofs is not None:
            raise NotImplementedError("timestep_cond and ofs are not supported.")
        if (image_rotary_emb is not None) != bool(self.config.use_rotary_positional_embeddings):
            raise ValueError(
                "image_rotary_emb must be given exactly when the model uses rotary positional embeddings "
                f"(use_rotary_positional_embeddings={self.config.use_rotary_positional_embeddings})."
            )

        sample_batch_size = hidden_states.size()[0]
        compiled_batch_size = self.compiled_batch_size
        if sample_batch_size != compiled_batch_size:
            raise ValueError(
                f"Mismatch between the transformer's runtime batch size ({sample_batch_size}) and compiled batch "
                f"size ({compiled_batch_size}). With guidance_scale > 1 the pipeline doubles the batch, so compile "
                "the transformer with batch_size = 2 * the pipeline batch size."
            )

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep])
        timestep = timestep.reshape(-1).expand(sample_batch_size).to(torch.float32)

        inputs = [
            hidden_states.to(self.dtype).contiguous(),
            encoder_hidden_states.to(self.dtype).contiguous(),
            timestep.contiguous(),
        ]
        if image_rotary_emb is not None:
            inputs += [freqs.to(torch.float32).contiguous() for freqs in image_rotary_emb]
        return super().forward(*inputs, return_dict=return_dict)


__all__ = [
    "RBLNCogVideoXTransformer3DModel",
]
