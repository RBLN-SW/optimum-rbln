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
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

from ...modeling_generic import RBLNModelForImageClassification
from .configuration_resnet import RBLNResNetForImageClassificationConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel


class RBLNResNetForImageClassification(RBLNModelForImageClassification):
    """
    RBLN optimized ResNet model for image classification tasks.

    This class provides hardware-accelerated inference for ResNet models
    on RBLN devices, supporting image classification with convolutional neural networks
    designed for computer vision tasks.
    """

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional["RBLNResNetForImageClassificationConfig"] = None,
    ) -> "RBLNResNetForImageClassificationConfig":
        if rbln_config.output_hidden_states is None:
            rbln_config.output_hidden_states = getattr(model_config, "output_hidden_states", False)

        rbln_config = super()._update_rbln_config(
            preprocessors=preprocessors,
            model=model,
            model_config=model_config,
            rbln_config=rbln_config,
        )

        return rbln_config

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: "RBLNResNetForImageClassificationConfig"
    ) -> torch.nn.Module:
        class _ResNetForImageClassification(torch.nn.Module):
            def __init__(self, model: torch.nn.Module, output_hidden_states: bool):
                super().__init__()
                self.model = model
                self.output_hidden_states = output_hidden_states

            def forward(self, *args, **kwargs):
                output = self.model(*args, output_hidden_states=self.output_hidden_states, **kwargs)
                return output

        return _ResNetForImageClassification(model, rbln_config.output_hidden_states)

    def forward(
        self, pixel_values: torch.Tensor, output_hidden_states: bool = None, return_dict: bool = None, **kwargs
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        """
        Foward pass for the RBLN-optimized ResNet model for image classification.

        Args:
            pixel_values (torch.FloatTensor of shape (batch_size, channels, height, width)): The tensors corresponding to the input images.
            output_hidden_states (bool, *optional*, defaults to False): Whether or not to return the hidden states of all layers.
                See hidden_states under returned tensors for more details.
            return_dict (bool, *optional*, defaults to True): Whether to return a dictionary of outputs.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a ImageClassifierOutputWithNoAttention object.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.rbln_config.output_hidden_states
        )

        if output_hidden_states != self.rbln_config.output_hidden_states:
            raise ValueError(
                f"Variable output_hidden_states {output_hidden_states} is not equal to rbln_config.output_hidden_states {self.rbln_config.output_hidden_states} "
                f"Please compile again with the correct argument."
            )

        return super().forward(pixel_values=pixel_values, return_dict=return_dict, **kwargs)
