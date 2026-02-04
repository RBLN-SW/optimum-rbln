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


from typing import TYPE_CHECKING, Tuple, Union

import torch
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput

from ...modeling_generic import RBLNModelForImageClassification


if TYPE_CHECKING:
    pass


class RBLNDetrForObjectDetection(RBLNModelForImageClassification):
    """
    RBLN optimized DETR model for object detection tasks.

    This class provides hardware-accelerated inference for DETR models
    on RBLN devices, supporting object detection with detection heads
    designed for object detection tasks.
    """

    def forward(
        self, pixel_values: torch.Tensor, return_dict: bool = None, **kwargs
    ) -> Union[Tuple, DetrObjectDetectionOutput]:
        """
        Foward pass for the RBLN-optimized DETR model for object detection.

        Args:
            pixel_values (torch.FloatTensor of shape (batch_size, channels, height, width)): The tensors corresponding to the input images.
            return_dict (bool, *optional*, defaults to True): Whether to return a dictionary of outputs.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a ImageClassifierOutputWithNoAttention object.
        """
        output = self.model[0](pixel_values=pixel_values, **kwargs)
        return DetrObjectDetectionOutput(
            logits=output[0], pred_boxes=output[1], last_hidden_state=output[2], encoder_last_hidden_state=output[3]
        )
