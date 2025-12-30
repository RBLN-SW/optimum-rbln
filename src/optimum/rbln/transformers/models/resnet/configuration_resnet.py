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


from typing import Optional

from ...configuration_generic import RBLNModelForImageClassificationConfig


class RBLNResNetForImageClassificationConfig(RBLNModelForImageClassificationConfig):
    """
    Configuration class for RBLNResNetForImageClassification.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized ResNet models for image classification tasks.
    """

    def __init__(self, output_hidden_states: Optional[bool] = None, **kwargs):
        """
        Args:
            image_size (Optional[Union[int, Tuple[int, int]]]): The size of input images.
                Can be an integer for square images or a tuple (height, width).
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            output_hidden_states (bool, optional) — Whether or not to return the hidden states of all layers.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.output_hidden_states = output_hidden_states
