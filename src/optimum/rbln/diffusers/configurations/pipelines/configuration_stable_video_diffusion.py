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

from typing import Any, ClassVar

from pydantic import Field, model_validator

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNCLIPVisionModelWithProjectionConfig
from ..models import RBLNAutoencoderKLTemporalDecoderConfig, RBLNUNetSpatioTemporalConditionModelConfig


class RBLNStableVideoDiffusionPipelineConfig(RBLNModelConfig):
    """Configuration for Stable Video Diffusion pipeline."""

    submodules: ClassVar[list[str]] = ["image_encoder", "unet", "vae"]
    _vae_uses_encoder: ClassVar[bool] = True

    image_encoder: dict[str, Any] | RBLNCLIPVisionModelWithProjectionConfig | None = Field(
        default=None, description="Configuration for the image encoder component."
    )
    unet: dict[str, Any] | RBLNUNetSpatioTemporalConditionModelConfig | None = Field(
        default=None, description="Configuration for the UNet model component."
    )
    vae: dict[str, Any] | RBLNAutoencoderKLTemporalDecoderConfig | None = Field(
        default=None, description="Configuration for the VAE model component."
    )

    # Pass-through parameters (excluded from serialization.)
    effective_batch_size: int | None = Field(
        default=None,
        alias="batch_size",
        exclude=True,
        description="Batch size for inference. Forwarded to image_encoder and vae.",
    )
    effective_height: int | None = Field(
        default=None,
        alias="height",
        exclude=True,
        description="Height of the generated video frames.",
    )
    effective_width: int | None = Field(
        default=None,
        alias="width",
        exclude=True,
        description="Width of the generated video frames.",
    )
    effective_num_frames: int | None = Field(
        default=None,
        alias="num_frames",
        exclude=True,
        description="Number of video frames to generate. Forwarded to unet and vae.",
    )
    effective_decode_chunk_size: int | None = Field(
        default=None,
        alias="decode_chunk_size",
        exclude=True,
        description="Number of frames to decode at a time in VAE. Forwarded to vae.",
    )
    effective_guidance_scale: float | None = Field(
        default=None,
        alias="guidance_scale",
        exclude=True,
        description="Scale for classifier-free guidance. Used to determine UNet batch size.",
    )

    @model_validator(mode="after")
    def initialize_submodules(self) -> "RBLNStableVideoDiffusionPipelineConfig":
        """Initialize submodule configs with pass-through parameters."""
        # Guard against re-entry during submodule initialization
        if getattr(self, "_submodules_initialized", False):
            return self
        object.__setattr__(self, "_submodules_initialized", True)

        height = self.effective_height
        width = self.effective_width

        if height is not None and width is not None:
            image_size = (height, width)
        else:
            # Get default image size from original class to set UNet, VAE image size
            height = self.get_default_values_for_original_cls("__call__", ["height"])["height"]
            width = self.get_default_values_for_original_cls("__call__", ["width"])["width"]
            image_size = (height, width)

        self.image_encoder = self.initialize_submodule_config(
            self.image_encoder,
            cls_name="RBLNCLIPVisionModelWithProjectionConfig",
            batch_size=self.effective_batch_size,
        )
        self.unet = self.initialize_submodule_config(
            self.unet,
            cls_name="RBLNUNetSpatioTemporalConditionModelConfig",
            num_frames=self.effective_num_frames,
        )
        self.vae = self.initialize_submodule_config(
            self.vae,
            cls_name="RBLNAutoencoderKLTemporalDecoderConfig",
            batch_size=self.effective_batch_size,
            num_frames=self.effective_num_frames,
            decode_chunk_size=self.effective_decode_chunk_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            sample_size=image_size,  # image size is equal to sample size in vae
        )

        # Get default guidance scale from original class to set UNet batch size
        guidance_scale = self.effective_guidance_scale
        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["max_guidance_scale"])[
                "max_guidance_scale"
            ]

        if not self.unet.batch_size_is_specified:
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.unet.batch_size = self.image_encoder.batch_size * 2
            else:
                self.unet.batch_size = self.image_encoder.batch_size

        return self

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def sample_size(self):
        return self.unet.sample_size

    @property
    def image_size(self):
        return self.vae.sample_size
