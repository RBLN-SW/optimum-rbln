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

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPVisionConfig, CLIPVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextModelOutput, CLIPVisionModelOutput

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from .configuration_clip import RBLNCLIPTextModelConfig, RBLNCLIPVisionModelConfig


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, CLIPTextModel, PreTrainedModel

    from ....diffusers.modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


class _TextEncoder(torch.nn.Module):
    def __init__(self, enc: "CLIPTextModel"):
        super().__init__()
        self.enc = enc

    def forward(self, inp):
        enc_out = self.enc(inp, output_hidden_states=True, return_dict=False)
        return enc_out


class RBLNCLIPTextModel(RBLNModel):
    """
    RBLN optimized CLIP text encoder model.

    This class provides hardware-accelerated inference for CLIP text encoders
    on RBLN devices, supporting text encoding for multimodal tasks.
    """

    _tp_support = False

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNCLIPTextModelConfig) -> torch.nn.Module:
        return _TextEncoder(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "CLIPTextConfig" = None,
        rbln_config: Optional[RBLNCLIPTextModelConfig] = None,
    ) -> RBLNCLIPTextModelConfig:
        input_info = [
            (
                "input_ids",
                [
                    rbln_config.batch_size,
                    model_config.max_position_embeddings,
                ],
                "int64",
            ),
        ]

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def forward(self, input_ids: torch.LongTensor, return_dict: Optional[bool] = None, **kwargs) -> torch.FloatTensor:
        """
        Forward pass for the RBLN-optimized CLIP text encoder model.

        Args:
            input_ids (torch.LongTensor): The input ids to the model.
            return_dict (Optional[bool]): Whether to return a dictionary of outputs.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a CLIPTextModelOutput object.
        """

        # To ignore using attention_mask, we override forward method.
        output = super().forward(input_ids, return_dict=return_dict)
        return output

    def _prepare_output(self, output, return_dict):
        # Prepare model output based on return_dict flag.
        # This method can be overridden by subclasses to provide task-specific output handling.

        if not return_dict:
            return (output,) if not isinstance(output, (tuple, list)) else output
        else:
            return CLIPTextModelOutput(
                text_embeds=output[0],
                last_hidden_state=output[1],
                hidden_states=output[2:],
            )


class RBLNCLIPTextModelWithProjection(RBLNCLIPTextModel):
    """
    RBLN optimized CLIP text encoder model with projection layer.

    This class extends RBLNCLIPTextModel with a projection layer for
    multimodal embedding alignment tasks.
    """


class _VisionEncoder(torch.nn.Module):
    def __init__(
        self,
        enc: CLIPVisionModel,
        interpolate_pos_encoding: bool,
        output_hidden_states: bool,
        output_attentions: bool,
    ):
        super().__init__()
        self.enc = enc
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions

    def forward(self, inp):
        enc_out = self.enc(
            inp,
            output_hidden_states=self.output_hidden_states,
            interpolate_pos_encoding=self.interpolate_pos_encoding,
            output_attentions=self.output_attentions,
            return_dict=False,
        )
        return enc_out


class RBLNCLIPVisionModel(RBLNModel):
    """
    RBLN optimized CLIP vision encoder model.

    This class provides hardware-accelerated inference for CLIP vision encoders
    on RBLN devices, supporting image encoding for multimodal tasks.
    """

    _tp_support = False

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNCLIPVisionModelConfig) -> torch.nn.Module:
        wrapper_cfg = {
            "interpolate_pos_encoding": rbln_config.interpolate_pos_encoding,
            "output_hidden_states": rbln_config.output_hidden_states,
            "output_attentions": rbln_config.output_attentions,
        }
        return _VisionEncoder(model, **wrapper_cfg).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "CLIPVisionConfig" = None,
        rbln_config: Optional[RBLNCLIPVisionModelConfig] = None,
    ) -> RBLNCLIPVisionModelConfig:
        if rbln_config.image_size is None:
            rbln_config.image_size = getattr(model_config, "image_size", None)

        if isinstance(rbln_config.image_size, int):
            rbln_config.image_size = (rbln_config.image_size, rbln_config.image_size)

        if rbln_config.image_size is None:
            raise ValueError("`rbln_image_size` should be specified!")

        if rbln_config.output_attentions is None:
            rbln_config.output_attentions = getattr(model_config, "output_attentions", False)

        if rbln_config.output_hidden_states is None:
            rbln_config.output_hidden_states = getattr(model_config, "output_hidden_states", False)

        rbln_compile_config = RBLNCompileConfig(
            input_info=[
                (
                    "pixel_values",
                    [
                        rbln_config.batch_size,
                        3,
                        rbln_config.image_height,
                        rbln_config.image_width,
                    ],
                    "float32",
                )
            ]
        )

        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        return_dict: bool = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Forward pass for the RBLN-optimized CLIP vision encoder model.

        Args:
            pixel_values (torch.Tensor): The pixel values to the model.
            return_dict (bool): Whether to return a dictionary of outputs.
            output_attentions (Optional[bool]): Whether to return attentions.
            output_hidden_states (Optional[bool]): Whether to return hidden states.
            interpolate_pos_encoding (bool): Whether to interpolate position encoding.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a BaseModelOutputWithPooling object.
        """

        if len(kwargs) > 0 and any(value is not None for value in kwargs.values()):
            logger.warning(
                f"Currently, optimum-rbln does not support kwargs {kwargs.keys()} for {self.__class__.__name__}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.rbln_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.rbln_config.output_hidden_states
        )

        if output_attentions != self.rbln_config.output_attentions:
            raise ValueError(
                f"Variable output_attentions {output_attentions} is not equal to rbln_config.output_attentions {self.rbln_config.output_attentions} "
                f"Please compile again with the correct argument."
            )

        if output_hidden_states != self.rbln_config.output_hidden_states:
            raise ValueError(
                f"Variable output_hidden_states {output_hidden_states} is not equal to rbln_config.output_hidden_states {self.rbln_config.output_hidden_states} "
                f"Please compile again with the correct argument."
            )

        if interpolate_pos_encoding != self.rbln_config.interpolate_pos_encoding:
            raise ValueError(
                f"Variable interpolate_pos_encoding {interpolate_pos_encoding} is not equal to rbln_config.interpolate_pos_encoding {self.rbln_config.interpolate_pos_encoding} "
                f"Please compile again with the correct argument."
            )

        output = super().forward(pixel_values, return_dict=return_dict)
        return output

    def _prepare_output(self, output, return_dict):
        # Prepare model output based on return_dict flag.
        # This method can be overridden by subclasses to provide task-specific output handling.
        last_hidden_state = output.pop(0)
        pooler_output = output.pop(0)
        vision_config = self.config.vision_config if hasattr(self.config, "vision_config") else self.config

        if self.rbln_config.output_hidden_states:
            hidden_states = ()
            num_hidden_layers = vision_config.num_hidden_layers
            for _ in range(num_hidden_layers + 1):
                hidden_states += (output.pop(0),)
        else:
            hidden_states = None

        if self.rbln_config.output_attentions:
            attentions = ()
            num_hidden_layers = vision_config.num_hidden_layers
            for _ in range(num_hidden_layers):
                attentions += (output.pop(0),)
        else:
            attentions = None

        if not return_dict:
            return tuple(
                item for item in (last_hidden_state, pooler_output, hidden_states, attentions) if item is not None
            )
        else:
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=hidden_states,
                attentions=attentions,
            )


class RBLNCLIPVisionModelWithProjection(RBLNCLIPVisionModel):
    """
    RBLN optimized CLIP vision encoder model with projection layer.

    This class extends RBLNCLIPVisionModel with a projection layer for
    multimodal embedding alignment tasks.
    """

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        return_dict: bool = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs,
    ) -> Union[Tuple, CLIPVisionModelOutput]:
        """
        Forward pass for the RBLN-optimized CLIP vision encoder model with projection.

        Args:
            pixel_values (torch.Tensor): The pixel values to the model.
            return_dict (bool): Whether to return a dictionary of outputs.
            output_attentions (Optional[bool]): Whether to return attentions.
            output_hidden_states (Optional[bool]): Whether to return hidden states.
            interpolate_pos_encoding (bool): Whether to interpolate position encoding.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a CLIPVisionModelOutput object.
        """

        return super().forward(
            pixel_values=pixel_values,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

    def _prepare_output(self, output, return_dict):
        # Prepare model output based on return_dict flag.
        # This method can be overridden by subclasses to provide task-specific output handling.

        image_embeds = output.pop(0) if isinstance(output, (tuple, list)) else output
        last_hidden_state = output.pop(0)

        vision_config = self.config.vision_config if hasattr(self.config, "vision_config") else self.config

        if self.rbln_config.output_hidden_states:
            hidden_states = ()
            num_hidden_layers = vision_config.num_hidden_layers
            for _ in range(num_hidden_layers + 1):
                hidden_states += (output.pop(0),)
        else:
            hidden_states = None

        if self.rbln_config.output_attentions:
            attentions = ()
            num_hidden_layers = vision_config.num_hidden_layers
            for _ in range(num_hidden_layers):
                attentions += (output.pop(0),)
        else:
            attentions = None

        if not return_dict:
            return tuple(
                item for item in (image_embeds, last_hidden_state, hidden_states, attentions) if item is not None
            )

        else:
            return CLIPVisionModelOutput(
                image_embeds=image_embeds,
                last_hidden_state=last_hidden_state,
                hidden_states=hidden_states,
                attentions=attentions,
            )
