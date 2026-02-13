# Copyright 2026 Rebellions Inc. All rights reserved.

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
from transformers import AutoModelForObjectDetection
from transformers.models.detr.modeling_detr import DetrConfig, DetrObjectDetectionOutput

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from .configuration_detr import RBLNDetrForObjectDetectionConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel


class RBLNDetrForObjectDetection(RBLNModel):
    """
    RBLN optimized DETR model for object detection tasks.

    This class provides hardware-accelerated inference for DETR models
    on RBLN devices, supporting object detection with detection heads
    designed for object detection tasks.
    """

    auto_model_class = AutoModelForObjectDetection

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "DetrConfig" = None,
        rbln_config: Optional[RBLNDetrForObjectDetectionConfig] = None,
    ) -> RBLNDetrForObjectDetectionConfig:
        if rbln_config.image_size is None:
            for processor in preprocessors:
                if hasattr(processor, "size"):
                    if all(required_key in processor.size.keys() for required_key in ["height", "width"]):
                        height, width = processor.size["height"], processor.size["width"]
                    elif "longest_edge" in processor.size.keys():
                        height, width = processor.size["longest_edge"], processor.size["longest_edge"]
                    break

            if rbln_config.image_size is None:
                raise ValueError("`image_size` should be specified!")
        else:
            if isinstance(rbln_config.image_size, int):
                height, width = rbln_config.image_size, rbln_config.image_size
            else:
                height, width = rbln_config.image_size

        rbln_compile_config = RBLNCompileConfig(
            input_info=[
                (
                    "pixel_values",
                    (rbln_config.batch_size, 3, height, width),
                    "float32",
                ),
                (
                    "pixel_mask",
                    (rbln_config.batch_size, height, width),
                    "int64",
                ),
            ]
        )
        rbln_config.image_size = (height, width)

        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
        return_dict: bool = None,
        **kwargs,
    ) -> Union[Tuple, DetrObjectDetectionOutput]:
        """
        Forward pass for the RBLN-optimized DETR model for object detection.

        Args:
            pixel_values (torch.FloatTensor of shape (batch_size, channels, height, width)): The tensors corresponding to the input images.
            pixel_mask (torch.LongTensor of shape (batch_size, height, width), optional): Mask from the image processor indicating valid pixels (1) vs padding (0).
            return_dict (bool, optional, defaults to True): Whether to return a dictionary of outputs.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors.
            Otherwise, returns a DetrObjectDetectionOutput object.
        """

        batch_size, _, height, width = pixel_values.shape
        max_height = self.rbln_config.image_size[0]
        max_width = self.rbln_config.image_size[1]

        pad_h = max_height - height
        pad_w = max_width - width

        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), dtype=torch.int64, device=pixel_values.device)

        pixel_values = torch.nn.functional.pad(pixel_values, (0, pad_w, 0, pad_h), value=0)
        pixel_mask = torch.nn.functional.pad(pixel_mask, (0, pad_w, 0, pad_h), value=0)

        output = self.model[0](pixel_values=pixel_values, pixel_mask=pixel_mask, **kwargs)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not return_dict:
            return output

        return DetrObjectDetectionOutput(
            logits=output[0],
            pred_boxes=output[1],
            last_hidden_state=output[2],
            encoder_last_hidden_state=output[3],
        )
