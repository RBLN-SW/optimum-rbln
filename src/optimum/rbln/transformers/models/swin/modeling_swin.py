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

import types
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import SwinConfig
from transformers.models.swin.modeling_swin import BackboneOutput

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from .configuration_swin import RBLNSwinBackboneConfig


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        PreTrainedModel,
        SwinBackbone,
        SwinEncoder,
    )


def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


def get_attn_mask(self, height, width, dtype, device):
    if self.shift_size > 0:
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, height, width, 1), dtype=dtype, device=device)
        height_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        width_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = torch.zeros(1)
        for height_slice in height_slices:
            for width_slice in width_slices:
                img_mask[:, height_slice, width_slice, :] = count
                count += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    else:
        attn_mask = None
    return attn_mask


class _SwinEncoder(torch.nn.Module):
    def __init__(self, model: "SwinEncoder"):
        super().__init__()
        self.layers = model.layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition
            )

            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                # rearrange b (h w) c -> b c h w
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = hidden_states_before_downsampling.view(
                    batch_size, *(output_dimensions[0], output_dimensions[1]), hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]

        return tuple(
            v
            for v in [hidden_states, all_hidden_states, all_self_attentions, all_reshaped_hidden_states]
            if v is not None
        )


class _SwinBackbone(torch.nn.Module):
    def __init__(self, model: "SwinBackbone", output_hidden_states: bool, output_attentions: bool):
        super().__init__()
        self.model = model
        self.embeddings = model.embeddings
        self.encoder = model.encoder
        self.stage_names = model.stage_names
        self.out_features = model.out_features
        self.hidden_states_norms = model.hidden_states_norms
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions

    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        embedding_output, input_dimensions = self.embeddings(pixel_values)
        outputs = _SwinEncoder(self.encoder)(
            embedding_output,
            input_dimensions,
            head_mask=None,
            output_attentions=self.output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            always_partition=True,
            return_dict=False,
        )

        hidden_states = outputs[-1]

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                batch_size, num_channels, height, width = hidden_state.shape
                hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
                hidden_state = hidden_state.view(batch_size, height * width, num_channels)
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                hidden_state = hidden_state.view(batch_size, height, width, num_channels)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps += (hidden_state,)

        output = (feature_maps,)

        if self.output_hidden_states:
            output += (outputs[1],)

        if self.output_attentions:
            output += (outputs[2],)

        return output


class RBLNSwinBackbone(RBLNModel):
    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNSwinBackboneConfig) -> torch.nn.Module:
        for layer in model.encoder.layers:
            for block in layer.blocks:
                block.get_attn_mask = types.MethodType(get_attn_mask, block)

        wrapper_cfg = {
            "output_hidden_states": rbln_config.output_hidden_states,
            "output_attentions": rbln_config.output_attentions,
        }
        return _SwinBackbone(model, **wrapper_cfg).eval()

    @classmethod
    def _update_submodule_config(
        cls,
        model: "PreTrainedModel",
        rbln_config: RBLNModelConfig,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]],
    ):
        for processor in preprocessors:
            if rbln_config.image_size is None and hasattr(processor, "image_processor"):
                if "height" in processor.image_processor.size and "width" in processor.image_processor.size:
                    rbln_config.image_size = (
                        processor.image_processor.size["height"],
                        processor.image_processor.size["width"],
                    )
                elif (
                    "longest_edge" in processor.image_processor.size
                    and "shortest_edge" in processor.image_processor.size
                ):
                    rbln_config.image_size = processor.image_processor.size["longest_edge"]
                elif "shortest_edge" in processor.image_processor.size:
                    rbln_config.image_size = processor.image_processor.size["shortest_edge"]
                break

        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "SwinConfig" = None,
        rbln_config: Optional[RBLNSwinBackboneConfig] = None,
    ) -> RBLNSwinBackboneConfig:
        if rbln_config.image_size is None:
            for processor in preprocessors:
                if hasattr(processor, "size"):
                    if all(required_key in processor.size.keys() for required_key in ["height", "width"]):
                        rbln_config.image_size = (processor.size["height"], processor.size["width"])
                    break

        input_info = [
            (
                "pixel_values",
                [
                    rbln_config.batch_size,
                    3,
                    rbln_config.image_height,
                    rbln_config.image_width,
                ],
                "float32",
            ),
        ]

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        **kwargs,
    ) -> Union[Tuple, BackboneOutput]:
        """
        Forward pass for the RBLN-optimized Swin backbone model.

        Args:
            pixel_values (torch.FloatTensor of shape (batch_size, num_channels, image_size, image_size), optional): The tensors corresponding to the input images. Pixel values can be obtained using ViTImageProcessor. See ViTImageProcessor.call() for details (processor_class uses ViTImageProcessor for processing images).
            return_dict (bool, optional): Whether or not to return a ModelOutput instead of a plain tuple.
            output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers. See attentions under returned tensors for more detail.
            output_hidden_states (bool, optional): Whether or not to return the hidden states of all layers. See hidden_states under returned tensors for more detail.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a BackboneOutput object.
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

        _, _, original_h, original_w = pixel_values.shape
        if original_h > self.rbln_config.image_height or original_w > self.rbln_config.image_width:
            raise ValueError(
                f"Input image size ({original_h}x{original_w}) exceeds the configured maximum size"
                f" ({self.rbln_config.image_height}x{self.rbln_config.image_width})."
            )

        pad_h = self.rbln_config.image_height - original_h
        pad_w = self.rbln_config.image_width - original_w
        padded_pixel_values = F.pad(pixel_values, (0, pad_w, 0, pad_h))

        output = self.model[0](padded_pixel_values)

        feature_maps = ()
        for _ in range(len(self.config.out_features)):
            feature_maps += (output.pop(0),)

        if self.rbln_config.output_hidden_states:
            hidden_states = ()
            for _ in range(len(self.config.stage_names)):
                hidden_states += (output.pop(0),)
        else:
            hidden_states = None

        if self.rbln_config.output_attentions:
            attentions = ()
            for _ in range(len(self.config.depths)):
                attentions += (output.pop(0),)
        else:
            attentions = None

        if not return_dict:
            return tuple(item for item in (feature_maps, hidden_states, attentions) if item is not None)
        else:
            return BackboneOutput(
                feature_maps=feature_maps,
                hidden_states=hidden_states,
                attentions=attentions,
            )
