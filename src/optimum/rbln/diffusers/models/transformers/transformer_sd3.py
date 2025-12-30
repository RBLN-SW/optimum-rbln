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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...configurations import RBLNSD3Transformer2DModelConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

logger = get_logger(__name__)


class SD3Transformer2DModelWrapper(torch.nn.Module):
    def __init__(self, model: "SD3Transformer2DModel") -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        # need controlnet support?
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        return self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            return_dict=False,
        )


class RBLNSD3Transformer2DModel(RBLNModel):
    """
    RBLN implementation of SD3Transformer2DModel for diffusion models like Stable Diffusion 3.

    The SD3Transformer2DModel takes text and/or image embeddings from encoders (like CLIP) and
    maps them to a shared latent space that guides the diffusion process to generate the desired image.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = SD3Transformer2DModel
    _output_class = Transformer2DModelOutput

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        return SD3Transformer2DModelWrapper(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        if rbln_config.sample_size is None:
            if rbln_config.image_size is not None:
                rbln_config.transformer.sample_size = (
                    rbln_config.image_size[0] // pipe.vae_scale_factor,
                    rbln_config.image_size[1] // pipe.vae_scale_factor,
                )
            else:
                rbln_config.transformer.sample_size = pipe.default_sample_size

        prompt_embed_length = pipe.tokenizer_max_length + rbln_config.max_seq_len
        rbln_config.transformer.prompt_embed_length = prompt_embed_length
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNSD3Transformer2DModelConfig,
    ) -> RBLNSD3Transformer2DModelConfig:
        if rbln_config.sample_size is None:
            rbln_config.sample_size = model_config.sample_size

        if isinstance(rbln_config.sample_size, int):
            rbln_config.sample_size = (rbln_config.sample_size, rbln_config.sample_size)

        input_info = [
            (
                "hidden_states",
                [
                    rbln_config.batch_size,
                    model_config.in_channels,
                    rbln_config.sample_size[0],
                    rbln_config.sample_size[1],
                ],
                "float32",
            ),
            (
                "encoder_hidden_states",
                [
                    rbln_config.batch_size,
                    rbln_config.prompt_embed_length,
                    model_config.joint_attention_dim,
                ],
                "float32",
            ),
            (
                "pooled_projections",
                [
                    rbln_config.batch_size,
                    model_config.pooled_projection_dim,
                ],
                "float32",
            ),
            ("timestep", [rbln_config.batch_size], "float32"),
        ]

        compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([compile_config])
        return rbln_config

    @property
    def compiled_batch_size(self):
        return self.rbln_config.compile_cfgs[0].input_info[0][1][0]

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        """
        Forward pass for the RBLN-optimized SD3Transformer2DModel.

        Args:
            hidden_states (torch.FloatTensor): The currently predicted image embeddings.
            encoder_hidden_states (torch.FloatTensor): Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (torch.FloatTensor): Embeddings projected from the embeddings of input conditions.
            timestep (torch.LongTensor): Current denoising step.
            return_dict (bool): Whether or not to return a [`~diffusers.models.modeling_output.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            (Union[`~diffusers.models.modeling_output.Transformer2DModelOutput`, Tuple])
        """
        sample_batch_size = hidden_states.size()[0]
        compiled_batch_size = self.compiled_batch_size
        if sample_batch_size != compiled_batch_size and (
            sample_batch_size * 2 == compiled_batch_size or sample_batch_size == compiled_batch_size * 2
        ):
            raise ValueError(
                f"Mismatch between transformer's runtime batch size ({sample_batch_size}) and compiled batch size ({compiled_batch_size}). "
                "This may be caused by the 'guidance scale' parameter, which doubles the runtime batch size in Stable Diffusion. "
                "Adjust the batch size of transformer during compilation.\n\n"
                "For details, see: https://docs.rbln.ai/software/optimum/model_api/diffusers/pipelines/stable_diffusion_3.html#important-batch-size-configuration-for-guidance-scale"
            )

        return super().forward(
            hidden_states, encoder_hidden_states, pooled_projections, timestep, return_dict=return_dict
        )
