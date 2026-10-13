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

import rebel
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel

from ....configuration_utils import DEFAULT_COMPILED_MODEL_NAME, RBLNCompileConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ....utils.logging import get_logger
from ...configurations import RBLNCogVideoXTransformer3DModelConfig


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

logger = get_logger(__name__)

# Upper bound of layers x batch x video tokens x width in one graph. Measured at 49 frames 480x720, batch 2 on 8
# devices: CogVideoX-2b (30 x 1920) runs as one graph in 4.1 s, CogVideoX-5b-I2V runs 21 of its 42 layers (x 3072)
# in 5.0 s, and all 42 in one graph fail with SYS_KERNEL_TIMEOUT on 8 or 16 devices.
MAX_GRAPH_WORK = 21 * 2 * 17550 * 3072


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


class CogVideoXTransformer3DModelSegment(torch.nn.Module):
    """
    Compile graph of one segment (the transformer blocks `start` to `end`) of CogVideoXTransformer3DModel.

    Used when the blocks do not fit in one graph (see `RBLNCogVideoXTransformer3DModelConfig.layers_per_graph`).
    The steps mirror `CogVideoXTransformer3DModel.forward` for CogVideoX 1.0 (no ofs, no temporal patches):

    - the first segment takes (hidden_states, encoder_hidden_states, timestep), runs the time and patch
      embeddings, and splits the text and video tokens;
    - the other segments take (hidden_states, encoder_hidden_states, emb) from the previous one;
    - the last segment adds the final norms, the projection and the unpatchify.

    With rotary positional embeddings, every segment also takes the RoPE (cos, sin) tables.
    """

    def __init__(self, model: CogVideoXTransformer3DModel, start: int, end: int, latent_shape: Sequence[int]):
        super().__init__()
        self.is_first = start == 0
        self.is_last = end == len(model.transformer_blocks)
        if self.is_first:
            self.time_proj = model.time_proj
            self.time_embedding = model.time_embedding
            self.patch_embed = model.patch_embed
            self.embedding_dropout = model.embedding_dropout
        self.transformer_blocks = torch.nn.ModuleList(model.transformer_blocks[start:end])
        if self.is_last:
            self.norm_final = model.norm_final
            self.norm_out = model.norm_out
            self.proj_out = model.proj_out
        self.patch_size = model.config.patch_size
        self.num_frames, self.height, self.width = latent_shape

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        emb: torch.Tensor,
        image_rotary_emb_cos: torch.Tensor | None = None,
        image_rotary_emb_sin: torch.Tensor | None = None,
    ):
        image_rotary_emb = None
        if image_rotary_emb_cos is not None:
            image_rotary_emb = (image_rotary_emb_cos, image_rotary_emb_sin)
        batch_size = hidden_states.shape[0]
        if self.is_first:
            # `emb` is the timestep here.
            t_emb = self.time_proj(emb).to(dtype=hidden_states.dtype)
            emb = self.time_embedding(t_emb, None)
            hidden_states = self.embedding_dropout(self.patch_embed(encoder_hidden_states, hidden_states))
            text_seq_length = encoder_hidden_states.shape[1]
            encoder_hidden_states = hidden_states[:, :text_seq_length]
            hidden_states = hidden_states[:, text_seq_length:]

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
            )

        if not self.is_last:
            return hidden_states, encoder_hidden_states, emb

        hidden_states = self.norm_final(hidden_states)
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)
        p = self.patch_size
        output = hidden_states.reshape(batch_size, self.num_frames, self.height // p, self.width // p, -1, p, p)
        return output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)


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

    @staticmethod
    def _segment_bounds(num_layers: int, layers_per_graph: int) -> list[tuple[int, int]]:
        num_graphs = -(-num_layers // layers_per_graph)
        # Even split, e.g. 42 layers with layers_per_graph=21 -> 2 graphs of 21.
        bounds = [round(i * num_layers / num_graphs) for i in range(num_graphs + 1)]
        return list(zip(bounds[:-1], bounds[1:], strict=True))

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNCogVideoXTransformer3DModelConfig
    ) -> torch.nn.Module | list[torch.nn.Module]:
        if getattr(model, "ofs_embedding", None) is not None or model.config.patch_size_t is not None:
            raise NotImplementedError(
                "RBLNCogVideoXTransformer3DModel supports CogVideoX 1.0 models (CogVideoX-2b, CogVideoX-5b-I2V); "
                "the ofs embedding and temporal patches of CogVideoX 1.5 are not supported."
            )
        bounds = cls._segment_bounds(model.config.num_layers, rbln_config.layers_per_graph)
        if len(bounds) == 1:
            return CogVideoXTransformer3DModelWrapper(model).eval()
        latent_shape = (rbln_config.num_latent_frames, *rbln_config.sample_size)
        return [CogVideoXTransformer3DModelSegment(model, start, end, latent_shape).eval() for start, end in bounds]

    @classmethod
    def get_compiled_model(
        cls, model: torch.nn.Module, rbln_config: RBLNCogVideoXTransformer3DModelConfig
    ) -> rebel.RBLNCompiledModel | dict[str, rebel.RBLNCompiledModel]:
        if len(rbln_config.compile_cfgs) == 1:
            return super().get_compiled_model(model, rbln_config)
        segments = cls._wrap_model_if_needed(model, rbln_config)
        return {
            cfg.compiled_model_name: cls.compile(
                segment,
                rbln_compile_config=cfg,
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device_map[cfg.compiled_model_name],
            )
            for segment, cfg in zip(segments, rbln_config.compile_cfgs, strict=True)
        }

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: list[rebel.RBLNCompiledModel],
        rbln_config: RBLNCogVideoXTransformer3DModelConfig,
    ) -> list[rebel.Runtime]:
        return [
            rebel.Runtime(
                compiled_model,
                tensor_type="pt",
                device=rbln_config.device_map[cfg.compiled_model_name],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for compiled_model, cfg in zip(compiled_models, rbln_config.compile_cfgs, strict=True)
        ]

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
        p = model_config.patch_size
        video_len = rbln_config.num_latent_frames * (latent_h // p) * (latent_w // p)
        inner_dim = model_config.num_attention_heads * model_config.attention_head_dim
        if rbln_config.layers_per_graph is None:
            # A graph fails on the device (SYS_KERNEL_TIMEOUT) once one run takes about 10 s. The run time grows
            # with the blocks x width x tokens in the graph; more devices than 8 do not lower it enough.
            work_per_layer = batch_size * video_len * inner_dim
            rbln_config.layers_per_graph = max(1, MAX_GRAPH_WORK // work_per_layer)

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
        rope_input_info = []
        if model_config.use_rotary_positional_embeddings:
            # The pipeline builds the 3D RoPE tables on the host (_prepare_rotary_positional_embeddings).
            rope_shape = [video_len, model_config.attention_head_dim]
            rope_input_info = [
                ("image_rotary_emb_cos", rope_shape, "float32"),
                ("image_rotary_emb_sin", rope_shape, "float32"),
            ]
        next_input_info = [
            ("hidden_states", [batch_size, video_len, inner_dim], rbln_config.dtype),
            ("encoder_hidden_states", [batch_size, rbln_config.max_text_seq_length, inner_dim], rbln_config.dtype),
            ("emb", [batch_size, model_config.time_embed_dim], rbln_config.dtype),
        ]

        bounds = cls._segment_bounds(model_config.num_layers, rbln_config.layers_per_graph)
        compile_cfgs = []
        for i, (start, end) in enumerate(bounds):
            name = DEFAULT_COMPILED_MODEL_NAME if len(bounds) == 1 else f"segment_{i}_layers_{start}_{end}"
            segment_input_info = (input_info if start == 0 else next_input_info) + rope_input_info
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name=name, input_info=segment_input_info))
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
        rope = []
        if image_rotary_emb is not None:
            rope = [freqs.to(torch.float32).contiguous() for freqs in image_rotary_emb]
        if len(self.model) == 1:
            return super().forward(*inputs, *rope, return_dict=return_dict)

        outputs = tuple(inputs)
        for runtime in self.model:
            outputs = runtime(*outputs, *rope)
        return self._prepare_output(outputs, return_dict)


__all__ = [
    "RBLNCogVideoXTransformer3DModel",
]
