# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, Optional, Tuple, Union

import torch

from ...configuration_generic import RBLNImageModelConfig

ExampleInputsType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _compute_sam3_spatial_shapes(
    image_height: int, image_width: int, patch_size: int
) -> List[Tuple[int, int]]:
    """Compute DETR decoder spatial shapes from image size and patch_size."""
    h = image_height // patch_size
    w = image_width // patch_size
    return [(h, w)]


class RBLNSam3ModelConfig(RBLNImageModelConfig):
    """Configuration for RBLNSam3Model.

    Values (image_size, patch_size, max_text_len) are resolved from model_config
    when loading from config.json.
    """

    subclass_non_save_attributes = ["example_inputs"]

    def __init__(
        self,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        batch_size: Optional[int] = None,
        max_text_len: Optional[int] = None,
        spatial_shapes_list: Optional[List[Tuple[int, int]]] = None,
        example_inputs: Optional[ExampleInputsType] = None,
        **kwargs: Any,
    ):
        super().__init__(image_size=image_size, batch_size=batch_size, **kwargs)
        self.max_text_len = max_text_len
        self.spatial_shapes_list = spatial_shapes_list
        self.example_inputs = example_inputs

    @property
    def spatial_shapes(self) -> torch.Tensor:
        """Spatial shapes tensor for DETR decoder (from spatial_shapes_list)."""
        if self.spatial_shapes_list is None:
            raise ValueError(
                "spatial_shapes_list is not set. Call _update_rbln_config before compile."
            )
        return torch.tensor(self.spatial_shapes_list, dtype=torch.long)
