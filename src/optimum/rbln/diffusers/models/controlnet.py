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

from typing import TYPE_CHECKING, Dict, Optional, Union

import torch
from diffusers import ControlNetModel
from diffusers.models.controlnets.controlnet import ControlNetOutput
from transformers import PretrainedConfig

from ...configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ...modeling import RBLNModel
from ...utils.logging import get_logger
from ...utils.model_utils import get_rbln_model_cls
from ..configurations import RBLNControlNetModelConfig
from ..modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel


logger = get_logger(__name__)


class _ControlNetModel(torch.nn.Module):
    def __init__(self, controlnet: "ControlNetModel"):
        super().__init__()
        self.controlnet = controlnet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale,
        text_embeds: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
    ):
        if text_embeds is not None and time_ids is not None:
            added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        else:
            added_cond_kwargs = {}

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=None,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        return down_block_res_samples, mid_block_res_sample


class _ControlNetModel_Cross_Attention(torch.nn.Module):
    def __init__(self, controlnet: "ControlNetModel"):
        super().__init__()
        self.controlnet = controlnet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale,
        text_embeds: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
    ):
        if text_embeds is not None and time_ids is not None:
            added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        else:
            added_cond_kwargs = {}

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )
        return down_block_res_samples, mid_block_res_sample


class RBLNControlNetModel(RBLNModel):
    """
    RBLN implementation of ControlNetModel for diffusion models.

    This model is used to accelerate ControlNetModel models from diffusers library on RBLN NPUs.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = ControlNetModel
    output_class = ControlNetOutput

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.use_encoder_hidden_states = any(
            item[0] == "encoder_hidden_states" for item in self.rbln_config.compile_cfgs[0].input_info
        )

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        use_encoder_hidden_states = False
        for down_block in model.down_blocks:
            if use_encoder_hidden_states := getattr(down_block, "has_cross_attention", False):
                break

        if use_encoder_hidden_states:
            return _ControlNetModel_Cross_Attention(model).eval()
        else:
            return _ControlNetModel(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: RBLNDiffusionMixin, rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        rbln_vae_cls = get_rbln_model_cls(f"RBLN{pipe.vae.__class__.__name__}")
        rbln_unet_cls = get_rbln_model_cls(f"RBLN{pipe.unet.__class__.__name__}")

        rbln_config.controlnet.max_seq_len = pipe.text_encoder.config.max_position_embeddings
        text_model_hidden_size = pipe.text_encoder_2.config.hidden_size if hasattr(pipe, "text_encoder_2") else None
        rbln_config.controlnet.text_model_hidden_size = text_model_hidden_size
        rbln_config.controlnet.vae_sample_size = rbln_vae_cls.get_vae_sample_size(pipe, rbln_config.vae)
        rbln_config.controlnet.unet_sample_size = rbln_unet_cls.get_unet_sample_size(
            pipe, rbln_config.unet, image_size=rbln_config.image_size
        )

        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNControlNetModelConfig,
    ) -> RBLNModelConfig:
        if rbln_config.unet_sample_size is None:
            raise ValueError("`unet_sample_size` (latent height, width) must be specified (ex. unet's sample_size)")

        if rbln_config.vae_sample_size is None:
            raise ValueError("`vae_sample_size` (input image height, width) must be specified (ex. vae's sample_size)")

        if rbln_config.max_seq_len is None:
            raise ValueError("`max_seq_len` (ex. text_encoder's max_position_embeddings) must be specified")

        input_info = [
            (
                "sample",
                [
                    rbln_config.batch_size,
                    model_config.in_channels,
                    rbln_config.unet_sample_size[0],
                    rbln_config.unet_sample_size[1],
                ],
                "float32",
            ),
            ("timestep", [], "float32"),
        ]

        use_encoder_hidden_states = any(element != "DownBlock2D" for element in model_config.down_block_types)
        if use_encoder_hidden_states:
            input_info.append(
                (
                    "encoder_hidden_states",
                    [rbln_config.batch_size, rbln_config.max_seq_len, model_config.cross_attention_dim],
                    "float32",
                )
            )

        input_info.append(
            (
                "controlnet_cond",
                [rbln_config.batch_size, 3, rbln_config.vae_sample_size[0], rbln_config.vae_sample_size[1]],
                "float32",
            )
        )
        input_info.append(("conditioning_scale", [], "float32"))

        if hasattr(model_config, "addition_embed_type") and model_config.addition_embed_type == "text_time":
            input_info.append(("text_embeds", [rbln_config.batch_size, rbln_config.text_model_hidden_size], "float32"))
            input_info.append(("time_ids", [rbln_config.batch_size, 6], "float32"))

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    @property
    def compiled_batch_size(self):
        return self.rbln_config.compile_cfgs[0].input_info[0][1][0]

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale: torch.Tensor = 1.0,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        """
        Forward pass for the RBLN-optimized ControlNetModel.

        Args:
            sample (torch.FloatTensor): The noisy input tensor.
            timestep (Union[torch.Tensor, float, int]): The number of timesteps to denoise an input.
            encoder_hidden_states (torch.Tensor): The encoder hidden states.
            controlnet_cond (torch.FloatTensor): The conditional input tensor of shape `(batch_size, max_seq_len, hidden_size)`.
            conditioning_scale (torch.Tensor): The scale factor for ControlNet outputs.
            added_cond_kwargs (Dict[str, torch.Tensor]): Additional conditions for the Stable Diffusion XL UNet.
            return_dict (bool): Whether or not to return a [`~diffusers.models.controlnets.controlnet.ControlNetOutput`] instead of a plain tuple

        Returns:
            (Union[`~diffusers.models.controlnets.controlnet.ControlNetOutput`], Tuple)
        """
        sample_batch_size = sample.size()[0]
        compiled_batch_size = self.compiled_batch_size
        if sample_batch_size != compiled_batch_size and (
            sample_batch_size * 2 == compiled_batch_size or sample_batch_size == compiled_batch_size * 2
        ):
            raise ValueError(
                f"Mismatch between ControlNet's runtime batch size ({sample_batch_size}) and compiled batch size ({compiled_batch_size}). "
                "This may be caused by the 'guidance_scale' parameter, which doubles the runtime batch size of ControlNet in Stable Diffusion. "
                "Adjust the batch size of ControlNet during compilation to match the runtime batch size.\n\n"
                "For details, see: https://docs.rbln.ai/software/optimum/model_api/diffusers/pipelines/controlnet.html#important-batch-size-configuration-for-guidance-scale"
            )

        added_cond_kwargs = {} if added_cond_kwargs is None else added_cond_kwargs
        if self.use_encoder_hidden_states:
            output = self.model[0](
                sample.contiguous(),
                timestep.float(),
                encoder_hidden_states,
                controlnet_cond,
                torch.tensor(conditioning_scale),
                **added_cond_kwargs,
            )
        else:
            output = self.model[0](
                sample.contiguous(),
                timestep.float(),
                controlnet_cond,
                torch.tensor(conditioning_scale),
                **added_cond_kwargs,
            )

        down_block_res_samples = output[:-1]
        mid_block_res_sample = output[-1]
        output = (down_block_res_samples, mid_block_res_sample)
        output = self._prepare_output(output, return_dict)
        return output

    def _prepare_output(self, output, return_dict):
        if not return_dict:
            return (output,) if not isinstance(output, (tuple, list)) else output
        else:
            return ControlNetOutput(
                down_block_res_samples=output[:-1],
                mid_block_res_sample=output[-1],
            )
