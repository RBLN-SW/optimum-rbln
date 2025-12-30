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

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import torch
from transformers import SiglipVisionConfig, SiglipVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...modeling_outputs import _validate_output_attentions, _validate_output_hidden_states
from .configuration_siglip import RBLNSiglipVisionModelConfig


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel


class _SiglipVisionModel(torch.nn.Module):
    def __init__(
        self,
        model: SiglipVisionModel,
        interpolate_pos_encoding: bool,
        output_hidden_states: bool,
        output_attentions: bool,
    ):
        super().__init__()
        self.vision_model = model.vision_model
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions

    def forward(self, inp):
        enc_out = self.vision_model(
            inp,
            output_hidden_states=self.output_hidden_states,
            return_dict=False,
            interpolate_pos_encoding=self.interpolate_pos_encoding,
            output_attentions=self.output_attentions,
        )
        return tuple(x for x in enc_out if x is not None)


class RBLNSiglipVisionModel(RBLNModel):
    """
    RBLN optimized SigLIP vision model.

    This class provides hardware-accelerated inference for SigLIP vision models
    on RBLN devices, supporting image encoding for multimodal vision-language tasks.
    """

    _tp_support = False

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNSiglipVisionModelConfig
    ) -> torch.nn.Module:
        wrapper_cfg = {
            "interpolate_pos_encoding": rbln_config.interpolate_pos_encoding,
            "output_hidden_states": rbln_config.output_hidden_states,
            "output_attentions": rbln_config.output_attentions,
        }
        return _SiglipVisionModel(model, **wrapper_cfg).eval()

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "SiglipVisionConfig" = None,
        rbln_config: Optional[RBLNSiglipVisionModelConfig] = None,
    ) -> RBLNSiglipVisionModelConfig:
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
        pixel_values: torch.Tensor,
        return_dict: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Any,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Forward pass for the RBLN-optimized SigLIP vision model.

        Args:
            pixel_values (torch.FloatTensor of shape (batch_size, num_channels, image_size, image_size), optional): The tensors corresponding to the input images. Pixel values can be obtained using ViTImageProcessor. See ViTImageProcessor.call() for details (processor_class uses ViTImageProcessor for processing images).
            return_dict (bool, optional): Whether or not to return a ModelOutput instead of a plain tuple.
            output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers. See attentions under returned tensors for more detail.
            output_hidden_states (bool, optional): Whether or not to return the hidden states of all layers. See hidden_states under returned tensors for more detail.
            interpolate_pos_encoding (bool, defaults to False): Whether to interpolate the pre-trained position encodings.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a BaseModelOutputWithPooling object.
        """

        output_attentions = _validate_output_attentions(output_attentions, self.rbln_config)
        output_hidden_states = _validate_output_hidden_states(output_hidden_states, self.rbln_config)
        if interpolate_pos_encoding != self.rbln_config.interpolate_pos_encoding:
            raise ValueError(
                f"Variable interpolate_pos_encoding {interpolate_pos_encoding} is not equal to rbln_config.interpolate_pos_encoding {self.rbln_config.interpolate_pos_encoding} "
                f"Please compile again with the correct argument."
            )

        output = super().forward(pixel_values, return_dict=return_dict, **kwargs)
        return output

    def _prepare_output(self, output, return_dict):
        # Prepare model output based on return_dict flag.
        # This method can be overridden by subclasses to provide task-specific output handling.

        if not return_dict:
            return (output,) if not isinstance(output, (tuple, list)) else output
        else:
            last_hidden_state = output.pop(0) if isinstance(output, (tuple, list)) else output
            vision_config = self.config.vision_config if hasattr(self.config, "vision_config") else self.config
            pooler_output = output.pop(0) if getattr(vision_config, "vision_use_head", True) else None

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

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=hidden_states,
                attentions=attentions,
            )
