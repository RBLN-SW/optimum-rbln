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
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...configurations import RBLNUNet2DConditionModelConfig
from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel

logger = get_logger(__name__)


class _UNet_SD(torch.nn.Module):
    def __init__(self, unet: "UNet2DConditionModel"):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        *down_and_mid_block_additional_residuals: Optional[Tuple[torch.Tensor]],
        text_embeds: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if text_embeds is not None and time_ids is not None:
            added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        else:
            added_cond_kwargs = {}

        if len(down_and_mid_block_additional_residuals) != 0:
            down_block_additional_residuals, mid_block_additional_residual = (
                down_and_mid_block_additional_residuals[:-1],
                down_and_mid_block_additional_residuals[-1],
            )
        else:
            down_block_additional_residuals, mid_block_additional_residual = None, None

        unet_out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        return unet_out


class _UNet_SDXL(torch.nn.Module):
    def __init__(self, unet: "UNet2DConditionModel"):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        *down_and_mid_block_additional_residuals: Optional[Tuple[torch.Tensor]],
    ) -> torch.Tensor:
        if len(down_and_mid_block_additional_residuals) == 2:
            added_cond_kwargs = {
                "text_embeds": down_and_mid_block_additional_residuals[0],
                "time_ids": down_and_mid_block_additional_residuals[1],
            }
            down_block_additional_residuals = None
            mid_block_additional_residual = None
        elif len(down_and_mid_block_additional_residuals) > 2:
            added_cond_kwargs = {
                "text_embeds": down_and_mid_block_additional_residuals[-2],
                "time_ids": down_and_mid_block_additional_residuals[-1],
            }
            down_block_additional_residuals, mid_block_additional_residual = (
                down_and_mid_block_additional_residuals[:-3],
                down_and_mid_block_additional_residuals[-3],
            )
        else:
            added_cond_kwargs = {}
            down_block_additional_residuals = None
            mid_block_additional_residual = None

        unet_out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        return unet_out


class _UNet_Kandinsky(torch.nn.Module):
    def __init__(self, unet: "UNet2DConditionModel"):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        added_cond_kwargs = {"image_embeds": image_embeds}

        unet_out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        return unet_out


class RBLNUNet2DConditionModel(RBLNModel):
    """
    RBLN implementation of UNet2DConditionModel for diffusion models.

    This model is used to accelerate UNet2DCondition models from diffusers library on RBLN NPUs.
    It is a key component in diffusion-based image generation models like Stable Diffusion.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = UNet2DConditionModel
    _rbln_config_class = RBLNUNet2DConditionModelConfig
    _output_class = UNet2DConditionOutput

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
        cls, model: torch.nn.Module, rbln_config: RBLNUNet2DConditionModelConfig
    ) -> torch.nn.Module:
        if model.config.addition_embed_type == "text_time":
            return _UNet_SDXL(model).eval()
        elif model.config.addition_embed_type == "image":
            return _UNet_Kandinsky(model).eval()
        else:
            return _UNet_SD(model).eval()

    @classmethod
    def get_unet_sample_size(
        cls,
        pipe: RBLNDiffusionMixin,
        rbln_config: RBLNUNet2DConditionModelConfig,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        if hasattr(pipe, "movq"):
            scale_factor = 2 ** (len(pipe.movq.config.block_out_channels) - 1)
        else:
            scale_factor = pipe.vae_scale_factor

        if image_size is None:
            if "Img2Img" in pipe.__class__.__name__:
                if hasattr(pipe, "vae"):
                    # In case of img2img, sample size of unet is determined by vae encoder.
                    vae_sample_size = pipe.vae.config.sample_size
                    if isinstance(vae_sample_size, int):
                        vae_sample_size = (vae_sample_size, vae_sample_size)

                    sample_size = (
                        vae_sample_size[0] // scale_factor,
                        vae_sample_size[1] // scale_factor,
                    )
                elif hasattr(pipe, "movq"):
                    logger.warning(
                        "RBLN config 'image_size' should have been provided for this pipeline. "
                        "Both variable will be set 512 by default."
                    )
                    sample_size = (512 // scale_factor, 512 // scale_factor)
            else:
                sample_size = pipe.unet.config.sample_size
                if isinstance(sample_size, int):
                    sample_size = (sample_size, sample_size)
        else:
            sample_size = (image_size[0] // scale_factor, image_size[1] // scale_factor)

        return sample_size

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: RBLNDiffusionMixin, rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        rbln_config.unet.text_model_hidden_size = (
            pipe.text_encoder_2.config.hidden_size if hasattr(pipe, "text_encoder_2") else None
        )
        rbln_config.unet.image_model_hidden_size = pipe.unet.config.encoder_hid_dim if hasattr(pipe, "unet") else None

        rbln_config.unet.max_seq_len = (
            pipe.text_encoder.config.max_position_embeddings if hasattr(pipe, "text_encoder") else None
        )

        rbln_config.unet.sample_size = cls.get_unet_sample_size(
            pipe, rbln_config.unet, image_size=rbln_config.image_size
        )
        rbln_config.unet.use_additional_residuals = "controlnet" in pipe.config.keys()

        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNUNet2DConditionModelConfig,
    ) -> RBLNUNet2DConditionModelConfig:
        if rbln_config.sample_size is None:
            rbln_config.sample_size = model_config.sample_size

        if isinstance(rbln_config.sample_size, int):
            rbln_config.sample_size = (rbln_config.sample_size, rbln_config.sample_size)

        input_info = [
            (
                "sample",
                [
                    rbln_config.batch_size,
                    model_config.in_channels,
                    rbln_config.sample_size[0],
                    rbln_config.sample_size[1],
                ],
                "float32",
            ),
            ("timestep", [], "float32"),
        ]

        if rbln_config.max_seq_len is not None:
            input_info.append(
                (
                    "encoder_hidden_states",
                    [rbln_config.batch_size, rbln_config.max_seq_len, model_config.cross_attention_dim],
                    "float32",
                ),
            )

        if rbln_config.use_additional_residuals:
            # down block addtional residuals
            first_shape = [
                rbln_config.batch_size,
                model_config.block_out_channels[0],
                rbln_config.sample_size[0],
                rbln_config.sample_size[1],
            ]
            height, width = rbln_config.sample_size[0], rbln_config.sample_size[1]
            input_info.append(("down_block_additional_residuals_0", first_shape, "float32"))
            name_idx = 1
            for idx, _ in enumerate(model_config.down_block_types):
                shape = [rbln_config.batch_size, model_config.block_out_channels[idx], height, width]
                for _ in range(model_config.layers_per_block):
                    input_info.append((f"down_block_additional_residuals_{name_idx}", shape, "float32"))
                    name_idx += 1
                if idx != len(model_config.down_block_types) - 1:
                    height = height // 2
                    width = width // 2
                    shape = [rbln_config.batch_size, model_config.block_out_channels[idx], height, width]
                    input_info.append((f"down_block_additional_residuals_{name_idx}", shape, "float32"))
                    name_idx += 1

            # mid block addtional residual
            num_cross_attn_blocks = model_config.down_block_types.count("CrossAttnDownBlock2D")
            out_channels = model_config.block_out_channels[-1]
            shape = [
                rbln_config.batch_size,
                out_channels,
                rbln_config.sample_size[0] // 2**num_cross_attn_blocks,
                rbln_config.sample_size[1] // 2**num_cross_attn_blocks,
            ]
            input_info.append(("mid_block_additional_residual", shape, "float32"))

        if hasattr(model_config, "addition_embed_type"):
            if model_config.addition_embed_type == "text_time":
                rbln_config.in_features = model_config.projection_class_embeddings_input_dim
                input_info.append(
                    ("text_embeds", [rbln_config.batch_size, rbln_config.text_model_hidden_size], "float32")
                )
                input_info.append(("time_ids", [rbln_config.batch_size, 6], "float32"))
            elif model_config.addition_embed_type == "image":
                input_info.append(
                    ("image_embeds", [rbln_config.batch_size, rbln_config.image_model_hidden_size], "float32")
                )

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
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        """
        Forward pass for the RBLN-optimized UNet2DConditionModel.

        Args:
            sample (torch.Tensor): The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (Union[torch.Tensor, float, int]): The number of timesteps to denoise an input.
            encoder_hidden_states (torch.Tensor): The encoder hidden states.
            added_cond_kwargs (Dict[str, torch.Tensor]): A kwargs dictionary containing additional embeddings that
                if specified are added to the embeddings that are passed along to the UNet blocks.
            down_block_additional_residuals (Optional[Tuple[torch.Tensor]]): A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual (Optional[torch.Tensor]): A tensor that if specified is added to the residual of the middle unet block.
            return_dict (bool): Whether or not to return a [`~diffusers.models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            (Union[`~diffusers.models.unets.unet_2d_condition.UNet2DConditionOutput`], Tuple)
        """
        sample_batch_size = sample.size()[0]
        compiled_batch_size = self.compiled_batch_size
        if sample_batch_size != compiled_batch_size and (
            sample_batch_size * 2 == compiled_batch_size or sample_batch_size == compiled_batch_size * 2
        ):
            raise ValueError(
                f"Mismatch between UNet's runtime batch size ({sample_batch_size}) and compiled batch size ({compiled_batch_size}). "
                "This may be caused by the 'guidance scale' parameter, which doubles the runtime batch size of UNet in Stable Diffusion. "
                "Adjust the batch size of UNet during compilation to match the runtime batch size.\n\n"
                "For details, see: https://docs.rbln.ai/software/optimum/model_api/diffusers/pipelines/stable_diffusion.html#important-batch-size-configuration-for-guidance-scale"
            )

        added_cond_kwargs = {} if added_cond_kwargs is None else added_cond_kwargs

        if down_block_additional_residuals is not None:
            down_block_additional_residuals = [t.contiguous() for t in down_block_additional_residuals]
            return super().forward(
                sample.contiguous(),
                timestep.float(),
                encoder_hidden_states,
                *down_block_additional_residuals,
                mid_block_additional_residual,
                **added_cond_kwargs,
                return_dict=return_dict,
            )

        if "image_embeds" in added_cond_kwargs:
            return super().forward(
                sample.contiguous(),
                timestep.float(),
                **added_cond_kwargs,
                return_dict=return_dict,
            )

        return super().forward(
            sample.contiguous(),
            timestep.float(),
            encoder_hidden_states,
            **added_cond_kwargs,
            return_dict=return_dict,
        )
