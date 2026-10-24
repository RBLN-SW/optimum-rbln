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

import copy
from contextlib import nullcontext

import torch
from diffusers.models.transformers.transformer_flux2 import (
    Flux2PosEmbed,
    Flux2Transformer2DModel,
    Flux2Transformer2DModelOutput,
)

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ...configurations.models.configuration_transformer_flux2 import RBLNFlux2Transformer2DModelConfig


def pack_rotary_embedding(position: Flux2PosEmbed, ids: torch.Tensor) -> torch.Tensor:
    # Native Flux2 shares positional coordinates across batch members.
    if ids.ndim == 3:
        ids = ids[0]
    cosine, sine = position(ids)
    return torch.cat((cosine, sine), dim=-1)


class _PackedRotaryEmbedding(torch.nn.Module):
    def forward(self, packed):
        return packed.chunk(2, dim=-1)


class Flux2TransformerWrapper(torch.nn.Module):
    """Retain native learned arithmetic while accepting native precomputed RoPE.

    Flux2's position module has no trained state and computes frequencies in
    float64. Its exact native CPU result is an input to the compiled graph.
    Only the copied position module is replaced; the full native transformer
    forward, all blocks and every learned parameter remain unchanged.
    """

    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.model.pos_embed = _PackedRotaryEmbedding()

    def forward(self, hidden_states, encoder_hidden_states, timestep, image_rotary_emb, text_rotary_emb):
        return self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=image_rotary_emb,
            txt_ids=text_rotary_emb,
            guidance=None,
            return_dict=False,
        )


class RBLNFlux2Transformer2DModel(RBLNModel):
    """Compiled FLUX.2 klein transformer with native positional arithmetic."""

    hf_library_name = "diffusers"
    auto_model_class = Flux2Transformer2DModel
    _rbln_config_class = RBLNFlux2Transformer2DModelConfig
    _output_class = Flux2Transformer2DModelOutput

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.pos_embed = Flux2PosEmbed(theta=self.config.rope_theta, axes_dim=self.config.axes_dims_rope)

    @classmethod
    def _wrap_model_if_needed(cls, model, rbln_config):
        if model.config.guidance_embeds:
            raise ValueError("This adapter requires a FLUX.2 klein checkpoint without guidance embeddings")
        return Flux2TransformerWrapper(model).eval()

    @classmethod
    def _update_rbln_config(cls, preprocessors, model, model_config, rbln_config):
        batch = rbln_config.batch_size
        images = rbln_config.image_token_count
        text = rbln_config.prompt_embed_length
        rotary_width = 2 * model_config.attention_head_dim
        rbln_config.set_compile_cfgs(
            [
                RBLNCompileConfig(
                    input_info=[
                        ("hidden_states", [batch, images, model_config.in_channels], rbln_config.dtype),
                        (
                            "encoder_hidden_states",
                            [batch, text, model_config.joint_attention_dim],
                            rbln_config.dtype,
                        ),
                        ("timestep", [batch], "float32"),
                        ("image_rotary_emb", [images, rotary_width], "float32"),
                        ("text_rotary_emb", [text, rotary_width], "float32"),
                    ]
                )
            ]
        )
        return rbln_config

    def cache_context(self, name):
        # The native pipeline enters named contexts without enabling a cache.
        return nullcontext()

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_ids,
        txt_ids,
        guidance=None,
        joint_attention_kwargs=None,
        return_dict=True,
    ):
        if guidance is not None or joint_attention_kwargs:
            raise ValueError("Guidance embeddings and runtime attention overrides are not compiled")
        return super().forward(
            hidden_states,
            encoder_hidden_states,
            timestep,
            pack_rotary_embedding(self.pos_embed, img_ids),
            pack_rotary_embedding(self.pos_embed, txt_ids),
            return_dict=return_dict,
        )


__all__ = ["RBLNFlux2Transformer2DModel"]
