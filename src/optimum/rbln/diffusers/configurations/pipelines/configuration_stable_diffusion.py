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

from pydantic import Field

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNCLIPTextModelConfig
from ..models import RBLNAutoencoderKLConfig, RBLNUNet2DConditionModelConfig


class RBLNStableDiffusionPipelineBaseConfig(RBLNModelConfig):
    """Base configuration for Stable Diffusion pipelines."""

    submodules: ClassVar[list[str]] = ["text_encoder", "unet", "vae"]
    _vae_uses_encoder: ClassVar[bool] = False
    _allow_no_compile_cfgs: ClassVar[bool] = True

    text_encoder: dict[str, Any] | RBLNModelConfig | None = Field(
        default=None, description="Configuration for the text encoder component."
    )
    unet: dict[str, Any] | RBLNModelConfig | None = Field(
        default=None, description="Configuration for the UNet model component."
    )
    vae: dict[str, Any] | RBLNModelConfig | None = Field(
        default=None, description="Configuration for the VAE model component."
    )

    def __init__(
        self,
        text_encoder: RBLNCLIPTextModelConfig | dict[str, Any] | None = None,
        unet: RBLNUNet2DConditionModelConfig | dict[str, Any] | None = None,
        vae: RBLNAutoencoderKLConfig | dict[str, Any] | None = None,
        *,
        batch_size: int | None = None,
        img_height: int | None = None,
        img_width: int | None = None,
        height: int | None = None,
        width: int | None = None,
        sample_size: tuple[int, int] | None = None,
        image_size: tuple[int, int] | None = None,
        guidance_scale: float | None = None,
        **kwargs: Any,
    ):
        # Initial check for image_size conflict remains as is
        if image_size is not None and (
            img_height is not None or img_width is not None or height is not None or width is not None
        ):
            raise ValueError("image_size cannot be provided alongside img_height/img_width or height/width")

        # Prioritize height/width (HF-aligned)
        if height is not None and width is not None:
            if img_height is not None or img_width is not None:
                raise ValueError(
                    "Cannot provide both 'height'/'width' and 'img_height'/'img_width' simultaneously. "
                    "Please use one set of arguments for image dimensions, preferring 'height'/'width'."
                )
            image_size = (height, width)
        elif (height is not None and width is None) or (height is None and width is not None):
            raise ValueError("Both height and width must be provided together if used")
        # Fallback to img_height/img_width for backward compatibility
        elif img_height is not None and img_width is not None:
            image_size = (img_height, img_width)
        elif (img_height is not None and img_width is None) or (img_height is None and img_width is not None):
            raise ValueError("Both img_height and img_width must be provided together if used")

        # Store submodule configs in kwargs for parent init
        kwargs["text_encoder"] = text_encoder
        kwargs["unet"] = unet
        kwargs["vae"] = vae

        super().__init__(**kwargs)

        # Initialize submodule configs
        self.text_encoder = self.initialize_submodule_config(
            self.text_encoder,
            cls_name="RBLNCLIPTextModelConfig",
            batch_size=batch_size,
        )
        self.unet = self.initialize_submodule_config(
            self.unet,
            cls_name="RBLNUNet2DConditionModelConfig",
            sample_size=sample_size,
        )
        self.vae = self.initialize_submodule_config(
            self.vae,
            cls_name="RBLNAutoencoderKLConfig",
            batch_size=batch_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            sample_size=image_size,
        )

        # Get default guidance scale from original class to set UNet batch size
        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]

        if isinstance(self.unet, RBLNModelConfig) and hasattr(self.unet, "batch_size_is_specified"):
            if not self.unet.batch_size_is_specified:
                do_classifier_free_guidance = guidance_scale > 1.0
                if do_classifier_free_guidance:
                    self.unet.batch_size = self.text_encoder.batch_size * 2
                else:
                    self.unet.batch_size = self.text_encoder.batch_size

    @property
    def batch_size(self) -> int:
        if isinstance(self.vae, RBLNModelConfig):
            return self.vae.batch_size
        return 1

    @property
    def sample_size(self) -> tuple[int, int] | None:
        if isinstance(self.unet, RBLNModelConfig):
            return self.unet.sample_size
        return None

    @property
    def image_size(self) -> tuple[int, int] | None:
        if isinstance(self.vae, RBLNModelConfig):
            return self.vae.sample_size
        return None


class RBLNStableDiffusionPipelineConfig(RBLNStableDiffusionPipelineBaseConfig):
    """
    Configuration for Stable Diffusion pipeline.
    """

    _vae_uses_encoder: ClassVar[bool] = False


class RBLNStableDiffusionImg2ImgPipelineConfig(RBLNStableDiffusionPipelineBaseConfig):
    """
    Configuration for Stable Diffusion image-to-image pipeline.
    """

    _vae_uses_encoder: ClassVar[bool] = True


class RBLNStableDiffusionInpaintPipelineConfig(RBLNStableDiffusionPipelineBaseConfig):
    """
    Configuration for Stable Diffusion inpainting pipeline.
    """

    _vae_uses_encoder: ClassVar[bool] = True
