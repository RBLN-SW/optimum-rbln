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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models.unets.unet_spatio_temporal_condition import (
    UNetSpatioTemporalConditionModel,
    UNetSpatioTemporalConditionOutput,
)
from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...configurations import RBLNUNetSpatioTemporalConditionModelConfig
from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, PreTrainedModel

logger = get_logger(__name__)


class _UNet_STCM(torch.nn.Module):
    def __init__(self, unet: "UNetSpatioTemporalConditionModel"):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
    ) -> torch.Tensor:
        unet_out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            return_dict=False,
        )
        return unet_out


class RBLNUNetSpatioTemporalConditionModel(RBLNModel):
    hf_library_name = "diffusers"
    auto_model_class = UNetSpatioTemporalConditionModel
    _rbln_config_class = RBLNUNetSpatioTemporalConditionModelConfig
    output_class = UNetSpatioTemporalConditionOutput
    output_key = "sample"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.in_features = self.rbln_config.in_features
        if self.in_features is not None:

            @dataclass
            class LINEAR1:
                in_features: int

            @dataclass
            class ADDEMBEDDING:
                linear_1: LINEAR1

            self.add_embedding = ADDEMBEDDING(LINEAR1(self.in_features))

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNUNetSpatioTemporalConditionModelConfig
    ) -> torch.nn.Module:
        return _UNet_STCM(model).eval()

    @classmethod
    def get_unet_sample_size(
        cls,
        pipe: RBLNDiffusionMixin,
        rbln_config: RBLNUNetSpatioTemporalConditionModelConfig,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Union[int, Tuple[int, int]]:
        scale_factor = pipe.vae_scale_factor

        if image_size is None:
            vae_sample_size = pipe.vae.config.sample_size
            if isinstance(vae_sample_size, int):
                vae_sample_size = (vae_sample_size, vae_sample_size)

            sample_size = (
                vae_sample_size[0] // scale_factor,
                vae_sample_size[1] // scale_factor,
            )
        else:
            sample_size = (image_size[0] // scale_factor, image_size[1] // scale_factor)
        return sample_size

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: RBLNDiffusionMixin, rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> Dict[str, Any]:
        rbln_config.unet.sample_size = cls.get_unet_sample_size(
            pipe, rbln_config.unet, image_size=rbln_config.image_size
        )
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNUNetSpatioTemporalConditionModelConfig,
    ) -> RBLNUNetSpatioTemporalConditionModelConfig:
        if rbln_config.num_frames is None:
            rbln_config.num_frames = model_config.num_frames

        if rbln_config.sample_size is None:
            rbln_config.sample_size = model_config.sample_size

        input_info = [
            (
                "sample",
                [
                    rbln_config.batch_size,
                    rbln_config.num_frames,
                    model_config.in_channels,
                    rbln_config.sample_size[0],
                    rbln_config.sample_size[1],
                ],
                "float32",
            ),
            ("timestep", [], "float32"),
            ("encoder_hidden_states", [rbln_config.batch_size, 1, model_config.cross_attention_dim], "float32"),
            ("added_time_ids", [rbln_config.batch_size, 3], "float32"),
        ]

        if hasattr(model_config, "addition_time_embed_dim"):
            rbln_config.in_features = model_config.projection_class_embeddings_input_dim

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])

        return rbln_config

    @property
    def compiled_batch_size(self):
        return self.rbln_config.compile_cfgs[0].input_info[0][1][0]

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        """
        Forward pass for the RBLN-optimized UNetSpatioTemporalConditionModel.

        Args:
            sample (torch.Tensor): The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (Union[torch.Tensor, float, int]): The number of timesteps to denoise an input.
            encoder_hidden_states (torch.Tensor): The encoder hidden states.
            added_time_ids (torch.Tensor): A tensor containing additional sinusoidal embeddings and added to the time embeddings.
            return_dict (bool): Whether or not to return a [`~diffusers.models.unets.unet_spatio_temporal_condition.UNetSpatioTemporalConditionOutput`] instead of a plain tuple.

        Returns:
            (Union[`~diffusers.models.unets.unet_spatio_temporal_condition.UNetSpatioTemporalConditionOutput`], Tuple)
        """
        sample_batch_size = sample.size()[0]
        compiled_batch_size = self.compiled_batch_size
        if sample_batch_size != compiled_batch_size and (
            sample_batch_size * 2 == compiled_batch_size or sample_batch_size == compiled_batch_size * 2
        ):
            raise ValueError(
                f"Mismatch between UNet's runtime batch size ({sample_batch_size}) and compiled batch size ({compiled_batch_size}). "
                "This may be caused by the 'guidance scale' parameter, which doubles the runtime batch size in Stable Diffusion. "
                "Adjust the batch size during compilation or modify the 'guidance scale' to match the compiled batch size.\n\n"
                "For details, see: https://docs.rbln.ai/software/optimum/model_api.html#stable-diffusion"
            )
        return super().forward(
            sample.contiguous(),
            timestep.float(),
            encoder_hidden_states,
            added_time_ids,
            return_dict=return_dict,
        )
