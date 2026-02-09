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
from transformers.modeling_outputs import ImageClassifierOutput

from ...modeling_generic import RBLNModelForImageClassification


class RBLNViTForImageClassification(RBLNModelForImageClassification):
    """
    RBLN optimized Vision Transformer (ViT) model for image classification tasks.

    This class provides hardware-accelerated inference for Vision Transformer models
    on RBLN devices, supporting image classification with transformer-based architectures
    that process images as sequences of patches.
    """

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> Union[ImageClassifierOutput, Tuple]:
        """
        Forward pass for the RBLN-optimized Vision Transformer model for image classification.

        Args:
            pixel_values (torch.FloatTensor of shape (batch_size, channels, height, width)):
                The tensors corresponding to the input images.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns an ImageClassifierOutput object.

        """
        return super().forward(pixel_values, **kwargs)
