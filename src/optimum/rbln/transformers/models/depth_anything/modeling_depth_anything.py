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


from typing import Tuple, Union

import torch
from transformers.modeling_outputs import DepthEstimatorOutput

from ...modeling_generic import RBLNModelForDepthEstimation


class RBLNDepthAnythingForDepthEstimation(RBLNModelForDepthEstimation):
    """
    RBLN optimized DepthAnythingForDepthEstimation model for depth estimation tasks.

    This class provides hardware-accelerated inference for Depth Anything V2
    models on RBLN devices, providing the most capable monocular depth estimation (MDE) model.
    """

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> Union[Tuple, DepthEstimatorOutput]:
        """
        Forward pass for the RBLN-optimized DepthAnythingForDepthEstimation model.

        Args:
            pixel_values (torch.FloatTensor of shape (batch_size, num_channels, height, width)): The tensors corresponding to the input images.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a DepthEstimatorOutput object.
        """
        return super().forward(pixel_values, **kwargs)
