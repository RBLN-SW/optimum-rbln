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

from ....configuration_utils import RBLNModelConfig


class RBLNStableDiffusionXLPipelineBaseConfig(RBLNModelConfig):
    """
    Base configuration for SDXL pipelines.

    Args:
        text_encoder: Configuration for the primary text encoder component.
        text_encoder_2: Configuration for the secondary text encoder component.
        unet: Configuration for the UNet model component.
        vae: Configuration for the VAE model component.
        img_height: Height of the generated images.
        img_width: Width of the generated images.
        height: Height of the generated images.
        width: Width of the generated images.
        sample_size: Spatial dimensions for the UNet model.
        _image_size: Internal image size (use height/width instead).
        guidance_scale: Scale for classifier-free guidance.
    """

    submodules: ClassVar[list[str]] = ["text_encoder", "text_encoder_2", "unet", "vae"]
    _vae_uses_encoder: ClassVar[bool] = False

    text_encoder: dict[str, Any] | RBLNModelConfig | None = None
    text_encoder_2: dict[str, Any] | RBLNModelConfig | None = None
    unet: dict[str, Any] | RBLNModelConfig | None = None
    vae: dict[str, Any] | RBLNModelConfig | None = None
    img_height: int | None = None
    img_width: int | None = None
    height: int | None = None
    width: int | None = None
    sample_size: tuple[int, int] | None = None
    guidance_scale: float | None = None

    def __init__(self, **data: Any):
        # Handle image_size kwarg if provided (before super().__init__)
        image_size_input = data.pop("image_size", None)
        batch_size = data.get("batch_size", None)

        super().__init__(**data)

        # Validate image size combinations
        if image_size_input is not None and (
            self.img_height is not None
            or self.img_width is not None
            or self.height is not None
            or self.width is not None
        ):
            raise ValueError("image_size cannot be provided alongside img_height/img_width or height/width")

        # Resolve image_size
        resolved_image_size = image_size_input
        if self.height is not None and self.width is not None:
            if self.img_height is not None or self.img_width is not None:
                raise ValueError(
                    "Cannot provide both 'height'/'width' and 'img_height'/'img_width' simultaneously. "
                    "Please use one set of arguments for image dimensions, preferring 'height'/'width'."
                )
            resolved_image_size = (self.height, self.width)
        elif (self.height is not None and self.width is None) or (self.height is None and self.width is not None):
            raise ValueError("Both height and width must be provided together if used")
        elif self.img_height is not None and self.img_width is not None:
            resolved_image_size = (self.img_height, self.img_width)
        elif (self.img_height is not None and self.img_width is None) or (
            self.img_height is None and self.img_width is not None
        ):
            raise ValueError("Both img_height and img_width must be provided together if used")

        # Initialize submodules
        if self.text_encoder is None or isinstance(self.text_encoder, dict):
            self.text_encoder = self.initialize_submodule_config(
                self.text_encoder,
                cls_name="RBLNCLIPTextModelConfig",
                batch_size=batch_size,
            )
        if self.text_encoder_2 is None or isinstance(self.text_encoder_2, dict):
            self.text_encoder_2 = self.initialize_submodule_config(
                self.text_encoder_2,
                cls_name="RBLNCLIPTextModelWithProjectionConfig",
                batch_size=batch_size,
            )
        if self.unet is None or isinstance(self.unet, dict):
            self.unet = self.initialize_submodule_config(
                self.unet,
                cls_name="RBLNUNet2DConditionModelConfig",
                sample_size=self.sample_size,
            )
        if self.vae is None or isinstance(self.vae, dict):
            self.vae = self.initialize_submodule_config(
                self.vae,
                cls_name="RBLNAutoencoderKLConfig",
                batch_size=batch_size,
                uses_encoder=self.__class__._vae_uses_encoder,
                sample_size=resolved_image_size,
            )

        # Get default guidance scale from original class to set UNet batch size
        guidance_scale = self.guidance_scale
        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]

        if not self.unet.batch_size_is_specified:
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.unet.batch_size = self.text_encoder.batch_size * 2
            else:
                self.unet.batch_size = self.text_encoder.batch_size

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def get_sample_size(self):
        return self.unet.sample_size

    @property
    def image_size(self):
        return self.vae.sample_size


class RBLNStableDiffusionXLPipelineConfig(RBLNStableDiffusionXLPipelineBaseConfig):
    """Config for SDXL Text2Img Pipeline"""

    _vae_uses_encoder = False


class RBLNStableDiffusionXLImg2ImgPipelineConfig(RBLNStableDiffusionXLPipelineBaseConfig):
    """Config for SDXL Img2Img Pipeline"""

    _vae_uses_encoder = True


class RBLNStableDiffusionXLInpaintPipelineConfig(RBLNStableDiffusionXLPipelineBaseConfig):
    """Config for SDXL Inpainting Pipeline"""

    _vae_uses_encoder = True
