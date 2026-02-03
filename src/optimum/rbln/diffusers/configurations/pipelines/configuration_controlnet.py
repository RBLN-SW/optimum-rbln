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

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNCLIPTextModelConfig, RBLNCLIPTextModelWithProjectionConfig
from ..models import RBLNAutoencoderKLConfig, RBLNControlNetModelConfig, RBLNUNet2DConditionModelConfig


class RBLNStableDiffusionControlNetPipelineBaseConfig(RBLNModelConfig):
    submodules: ClassVar[list[str]] = ["text_encoder", "unet", "vae", "controlnet"]
    _vae_uses_encoder: ClassVar[bool] = False

    text_encoder: dict[str, Any] | RBLNCLIPTextModelConfig | None = None
    unet: dict[str, Any] | RBLNUNet2DConditionModelConfig | None = None
    vae: dict[str, Any] | RBLNAutoencoderKLConfig | None = None
    controlnet: dict[str, Any] | RBLNControlNetModelConfig | None = None

    def __init__(
        self,
        *,
        batch_size: Optional[int] = None,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        sample_size: Optional[Tuple[int, int]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        guidance_scale: Optional[float] = None,
        **data: Any,
    ):
        """
        Args:
            text_encoder (Optional[RBLNCLIPTextModelConfig]): Configuration for the text encoder component.
                Initialized as RBLNCLIPTextModelConfig if not provided.
            unet (Optional[RBLNUNet2DConditionModelConfig]): Configuration for the UNet model component.
                Initialized as RBLNUNet2DConditionModelConfig if not provided.
            vae (Optional[RBLNAutoencoderKLConfig]): Configuration for the VAE model component.
                Initialized as RBLNAutoencoderKLConfig if not provided.
            controlnet (Optional[RBLNControlNetModelConfig]): Configuration for the ControlNet model component.
                Initialized as RBLNControlNetModelConfig if not provided.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            img_height (Optional[int]): Height of the generated images.
            img_width (Optional[int]): Width of the generated images.
            height (Optional[int]): Height of the generated images.
            width (Optional[int]): Width of the generated images.
            sample_size (Optional[Tuple[int, int]]): Spatial dimensions for the UNet model.
            image_size (Optional[Tuple[int, int]]): Alternative way to specify image dimensions.
                Cannot be used together with img_height/img_width.
            guidance_scale (Optional[float]): Scale for classifier-free guidance.
            **data: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If both image_size and img_height/img_width are provided.

        Note:
            When guidance_scale > 1.0, the UNet batch size is automatically doubled to
            accommodate classifier-free guidance.
        """
        super().__init__(**data)

        # Initial check for image_size conflict remains as is
        if image_size is not None and (
            img_height is not None or img_width is not None or height is not None or width is not None
        ):
            raise ValueError("image_size cannot be provided alongside img_height/img_width or height/width")

        # Prioritize height/width (HF-aligned)
        if height is not None and width is not None:
            if img_height is not None or img_width is not None:
                # Raise error if both sets of arguments are provided
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
            sample_size=image_size,  # image size is equal to sample size in vae
        )
        self.controlnet = self.initialize_submodule_config(
            self.controlnet,
            cls_name="RBLNControlNetModelConfig",
        )

        # Get default guidance scale from original class to set UNet and ControlNet batch size
        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]

        if guidance_scale is not None:
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                if not self.unet.batch_size_is_specified:
                    self.unet.batch_size = self.text_encoder.batch_size * 2
                if not self.controlnet.batch_size_is_specified:
                    self.controlnet.batch_size = self.text_encoder.batch_size * 2
            else:
                if not self.unet.batch_size_is_specified:
                    self.unet.batch_size = self.text_encoder.batch_size
                if not self.controlnet.batch_size_is_specified:
                    self.controlnet.batch_size = self.text_encoder.batch_size

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def sample_size(self):
        return self.unet.sample_size

    @property
    def image_size(self):
        return self.vae.sample_size


class RBLNStableDiffusionControlNetPipelineConfig(RBLNStableDiffusionControlNetPipelineBaseConfig):
    """
    Configuration for Stable Diffusion ControlNet pipeline.
    """

    _vae_uses_encoder: ClassVar[bool] = False


class RBLNStableDiffusionControlNetImg2ImgPipelineConfig(RBLNStableDiffusionControlNetPipelineBaseConfig):
    """
    Configuration for Stable Diffusion ControlNet image-to-image pipeline.
    """

    _vae_uses_encoder: ClassVar[bool] = True


class RBLNStableDiffusionXLControlNetPipelineBaseConfig(RBLNModelConfig):
    """
    Base configuration for Stable Diffusion XL ControlNet pipelines.
    """

    submodules: ClassVar[list[str]] = ["text_encoder", "text_encoder_2", "unet", "vae", "controlnet"]
    _vae_uses_encoder: ClassVar[bool] = False

    text_encoder: dict[str, Any] | RBLNCLIPTextModelConfig | None = None
    text_encoder_2: dict[str, Any] | RBLNCLIPTextModelWithProjectionConfig | None = None
    unet: dict[str, Any] | RBLNUNet2DConditionModelConfig | None = None
    vae: dict[str, Any] | RBLNAutoencoderKLConfig | None = None
    controlnet: dict[str, Any] | RBLNControlNetModelConfig | None = None

    def __init__(
        self,
        *,
        batch_size: Optional[int] = None,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        sample_size: Optional[Tuple[int, int]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        guidance_scale: Optional[float] = None,
        **data: Any,
    ):
        """
        Args:
            text_encoder (Optional[RBLNCLIPTextModelConfig]): Configuration for the primary text encoder.
                Initialized as RBLNCLIPTextModelConfig if not provided.
            text_encoder_2 (Optional[RBLNCLIPTextModelWithProjectionConfig]): Configuration for the secondary text encoder.
                Initialized as RBLNCLIPTextModelWithProjectionConfig if not provided.
            unet (Optional[RBLNUNet2DConditionModelConfig]): Configuration for the UNet model component.
                Initialized as RBLNUNet2DConditionModelConfig if not provided.
            vae (Optional[RBLNAutoencoderKLConfig]): Configuration for the VAE model component.
                Initialized as RBLNAutoencoderKLConfig if not provided.
            controlnet (Optional[RBLNControlNetModelConfig]): Configuration for the ControlNet model component.
                Initialized as RBLNControlNetModelConfig if not provided.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            img_height (Optional[int]): Height of the generated images.
            img_width (Optional[int]): Width of the generated images.
            height (Optional[int]): Height of the generated images.
            width (Optional[int]): Width of the generated images.
            sample_size (Optional[Tuple[int, int]]): Spatial dimensions for the UNet model.
            image_size (Optional[Tuple[int, int]]): Alternative way to specify image dimensions.
                Cannot be used together with img_height/img_width.
            guidance_scale (Optional[float]): Scale for classifier-free guidance.
            **data: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If both image_size and img_height/img_width are provided.

        Note:
            When guidance_scale > 1.0, the UNet batch size is automatically doubled to
            accommodate classifier-free guidance.
        """
        super().__init__(**data)

        # Initial check for image_size conflict remains as is
        if image_size is not None and (
            img_height is not None or img_width is not None or height is not None or width is not None
        ):
            raise ValueError("image_size cannot be provided alongside img_height/img_width or height/width")

        # Prioritize height/width (HF-aligned)
        if height is not None and width is not None:
            if img_height is not None or img_width is not None:
                # Raise error if both sets of arguments are provided
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

        self.text_encoder = self.initialize_submodule_config(
            self.text_encoder,
            cls_name="RBLNCLIPTextModelConfig",
            batch_size=batch_size,
        )
        self.text_encoder_2 = self.initialize_submodule_config(
            self.text_encoder_2,
            cls_name="RBLNCLIPTextModelWithProjectionConfig",
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
            sample_size=image_size,  # image size is equal to sample size in vae
        )
        self.controlnet = self.initialize_submodule_config(
            self.controlnet,
            cls_name="RBLNControlNetModelConfig",
        )

        # Get default guidance scale from original class to set UNet and ControlNet batch size
        guidance_scale = (
            guidance_scale
            or self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]
        )

        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            if not self.unet.batch_size_is_specified:
                self.unet.batch_size = self.text_encoder.batch_size * 2
            if not self.controlnet.batch_size_is_specified:
                self.controlnet.batch_size = self.text_encoder.batch_size * 2
        else:
            if not self.unet.batch_size_is_specified:
                self.unet.batch_size = self.text_encoder.batch_size
            if not self.controlnet.batch_size_is_specified:
                self.controlnet.batch_size = self.text_encoder.batch_size

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def sample_size(self):
        return self.unet.sample_size

    @property
    def image_size(self):
        return self.vae.sample_size


class RBLNStableDiffusionXLControlNetPipelineConfig(RBLNStableDiffusionXLControlNetPipelineBaseConfig):
    """
    Configuration for Stable Diffusion XL ControlNet pipeline.
    """

    _vae_uses_encoder: ClassVar[bool] = False


class RBLNStableDiffusionXLControlNetImg2ImgPipelineConfig(RBLNStableDiffusionXLControlNetPipelineBaseConfig):
    """
    Configuration for Stable Diffusion XL ControlNet image-to-image pipeline.
    """

    _vae_uses_encoder: ClassVar[bool] = True
