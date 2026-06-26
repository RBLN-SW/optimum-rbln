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

from typing import Any, Optional, Tuple

from ....configuration_utils import RBLNAutoConfig, RBLNModelConfig
from ....transformers import RBLNSiglipVisionModelConfig


class RBLNVideoSafetyModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Video Content Safety Filter.
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        input_size: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.input_size = input_size or 1152


class RBLNRetinaFaceFilterConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Retina Face Filter.
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.image_size = image_size or (704, 1280)


class RBLNCosmosSafetyCheckerConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Cosmos Safety Checker.
    """

    submodules = ["llamaguard3", "video_safety_model", "face_blur_filter", "siglip_encoder"]

    def __init__(
        self,
        llamaguard3: Optional[RBLNModelConfig] = None,
        video_safety_model: Optional[RBLNModelConfig] = None,
        face_blur_filter: Optional[RBLNModelConfig] = None,
        siglip_encoder: Optional[RBLNSiglipVisionModelConfig] = None,
        *,
        batch_size: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if height is not None and width is not None:
            image_size = (height, width)

        if max_seq_len is None:
            max_seq_len = 512

        tensor_parallel_size = kwargs.get("tensor_parallel_size")

        self.llamaguard3 = self.initialize_submodule_config(
            llamaguard3,
            cls_name="RBLNLlamaForCausalLMConfig",
            batch_size=batch_size,
            tensor_parallel_size=tensor_parallel_size,
            max_seq_len=max_seq_len,
        )
        self.siglip_encoder = self.initialize_submodule_config(
            siglip_encoder,
            cls_name="RBLNSiglipVisionModelConfig",
            batch_size=batch_size,
            image_size=(384, 384),
        )
        self.video_safety_model = self.initialize_submodule_config(
            video_safety_model,
            cls_name="RBLNVideoSafetyModelConfig",
            batch_size=batch_size,
            input_size=1152,
        )
        self.face_blur_filter = self.initialize_submodule_config(
            face_blur_filter,
            cls_name="RBLNRetinaFaceFilterConfig",
            batch_size=batch_size,
            image_size=image_size,
        )


RBLNAutoConfig.register(RBLNVideoSafetyModelConfig)
RBLNAutoConfig.register(RBLNRetinaFaceFilterConfig)
RBLNAutoConfig.register(RBLNCosmosSafetyCheckerConfig)
