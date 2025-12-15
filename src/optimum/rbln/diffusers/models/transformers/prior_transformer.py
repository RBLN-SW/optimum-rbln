# Copyright 2024 Rebellions Inc.

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
from typing import TYPE_CHECKING, Optional, Union

import torch
from diffusers.models.transformers.prior_transformer import PriorTransformer, PriorTransformerOutput

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...configurations.models import RBLNPriorTransformerConfig
from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

logger = get_logger(__name__)


class _PriorTransformer(torch.nn.Module):
    def __init__(self, prior: PriorTransformer):
        super().__init__()
        self._prior = prior

    def forward(
        self,
        hidden_states,
        timestep,
        proj_embedding,
        encoder_hidden_states,
        attention_mask,
        return_dict=True,
    ):
        return self._prior.forward(
            hidden_states,
            timestep,
            proj_embedding,
            encoder_hidden_states,
            attention_mask,
            return_dict=False,
        )


class RBLNPriorTransformer(RBLNModel):
    """
    RBLN implementation of PriorTransformer for diffusion models like Kandinsky V2.2.

    The PriorTransformer takes text and/or image embeddings from encoders (like CLIP) and
    maps them to a shared latent space that guides the diffusion process to generate the desired image.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = PriorTransformer
    _output_class = PriorTransformerOutput

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.clip_mean = artifacts["clip_mean"]
        self.clip_std = artifacts["clip_std"]

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        return _PriorTransformer(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: RBLNDiffusionMixin, rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        return rbln_config

    @classmethod
    def save_torch_artifacts(
        cls, model: "PreTrainedModel", save_dir_path: Path, subfolder: str, rbln_config: RBLNModelConfig
    ):
        save_dict = {}
        save_dict["clip_mean"] = model.clip_mean
        save_dict["clip_std"] = model.clip_std
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNPriorTransformerConfig,
    ) -> RBLNPriorTransformerConfig:
        rbln_config.embedding_dim = rbln_config.embedding_dim or model_config.embedding_dim
        rbln_config.num_embeddings = rbln_config.num_embeddings or model_config.num_embeddings

        input_info = [
            ("hidden_states", [rbln_config.batch_size, rbln_config.embedding_dim], "float32"),
            ("timestep", [], "float32"),
            ("proj_embedding", [rbln_config.batch_size, rbln_config.embedding_dim], "float32"),
            (
                "encoder_hidden_states",
                [rbln_config.batch_size, rbln_config.num_embeddings, rbln_config.embedding_dim],
                "float32",
            ),
            ("attention_mask", [rbln_config.batch_size, rbln_config.num_embeddings], "float32"),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def post_process_latents(self, prior_latents):
        prior_latents = (prior_latents * self.clip_std) + self.clip_mean
        return prior_latents

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        proj_embedding: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Forward pass for the RBLN-optimized PriorTransformer.

        Args:
            hidden_states (torch.Tensor): The currently predicted image embeddings.
            timestep (Union[torch.Tensor, float, int]): Current denoising step.
            proj_embedding (torch.Tensor): Projected embedding vector the denoising process is conditioned on.
            encoder_hidden_states (Optional[torch.Tensor]): Hidden states of the text embeddings the denoising process is conditioned on.
            attention_mask (Optional[torch.Tensor]): Text mask for the text embeddings.
            return_dict (bool): Whether or not to return a [`~diffusers.models.transformers.prior_transformer.PriorTransformerOutput`] instead of a plain tuple.

        Returns:
            (Union[`~diffusers.models.transformers.prior_transformer.PriorTransformerOutput`, Tuple])
        """
        # Convert timestep(long) and attention_mask(bool) to float
        return super().forward(
            hidden_states,
            timestep.float(),
            proj_embedding,
            encoder_hidden_states,
            attention_mask.float(),
            return_dict=return_dict,
        )
