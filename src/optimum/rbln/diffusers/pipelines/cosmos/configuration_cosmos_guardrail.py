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

from __future__ import annotations

from typing import Any, ClassVar, Optional, Tuple

from pydantic import Field, field_validator

from ....configuration_utils import RBLNAutoConfig, RBLNModelConfig
from ....transformers import RBLNSiglipVisionModelConfig


class RBLNVideoSafetyModelConfig(RBLNModelConfig):
    """Configuration class for RBLN Video Content Safety Filter."""

    batch_size: int = Field(default=1, description="The batch size for inference.")
    input_size: int = Field(default=1152, description="Input feature size for the safety model.")

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        return v if v is not None else 1

    @field_validator("input_size", mode="before")
    @classmethod
    def validate_input_size(cls, v: int | None) -> int:
        return v if v is not None else 1152


class RBLNRetinaFaceFilterConfig(RBLNModelConfig):
    """Configuration class for RBLN Retina Face Filter."""

    batch_size: int = Field(default=1, description="The batch size for inference.")
    image_size: tuple[int, int] = Field(
        default=(704, 1280), description="The size of input images as (height, width)."
    )

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        return v if v is not None else 1

    @field_validator("image_size", mode="before")
    @classmethod
    def validate_image_size(cls, v: tuple[int, int] | None) -> tuple[int, int]:
        return v if v is not None else (704, 1280)


class RBLNCosmosSafetyCheckerConfig(RBLNModelConfig):
    """Configuration class for RBLN Cosmos Safety Checker."""

    submodules: ClassVar[list[str]] = ["llamaguard3", "video_safety_model", "face_blur_filter", "siglip_encoder"]

    llamaguard3: dict[str, Any] | RBLNModelConfig | None = Field(
        default=None, description="Configuration for the LlamaGuard3 safety model."
    )
    video_safety_model: dict[str, Any] | RBLNModelConfig | None = Field(
        default=None, description="Configuration for the video safety model."
    )
    face_blur_filter: dict[str, Any] | RBLNModelConfig | None = Field(
        default=None, description="Configuration for the face blur filter."
    )
    siglip_encoder: dict[str, Any] | RBLNSiglipVisionModelConfig | None = Field(
        default=None, description="Configuration for the SigLIP vision encoder."
    )

    def __init__(
        self,
        *,
        batch_size: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        **data: Any,
    ):
        super().__init__(**data)
        if height is not None and width is not None:
            image_size = (height, width)

        if max_seq_len is None:
            max_seq_len = 512

        tensor_parallel_size = data.get("tensor_parallel_size")

        self.llamaguard3 = self.initialize_submodule_config(
            self.llamaguard3,
            cls_name="RBLNLlamaForCausalLMConfig",
            batch_size=batch_size,
            tensor_parallel_size=tensor_parallel_size,
            max_seq_len=max_seq_len,
        )
        self.siglip_encoder = self.initialize_submodule_config(
            self.siglip_encoder,
            cls_name="RBLNSiglipVisionModelConfig",
            batch_size=batch_size,
            image_size=(384, 384),
        )
        self.video_safety_model = self.initialize_submodule_config(
            self.video_safety_model,
            cls_name="RBLNVideoSafetyModelConfig",
            batch_size=batch_size,
            input_size=1152,
        )
        self.face_blur_filter = self.initialize_submodule_config(
            self.face_blur_filter,
            cls_name="RBLNRetinaFaceFilterConfig",
            batch_size=batch_size,
            image_size=image_size,
        )


RBLNAutoConfig.register(RBLNVideoSafetyModelConfig)
RBLNAutoConfig.register(RBLNRetinaFaceFilterConfig)
RBLNAutoConfig.register(RBLNCosmosSafetyCheckerConfig)
