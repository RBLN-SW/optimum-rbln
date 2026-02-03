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
from ....transformers import RBLNCLIPTextModelWithProjectionConfig, RBLNCLIPVisionModelWithProjectionConfig
from ..models import RBLNUNet2DConditionModelConfig, RBLNVQModelConfig
from ..models.configuration_prior_transformer import RBLNPriorTransformerConfig


class RBLNKandinskyV22PipelineBaseConfig(RBLNModelConfig):
    submodules: ClassVar[list[str]] = ["unet", "movq"]
    _movq_uses_encoder: ClassVar[bool] = False

    unet: dict[str, Any] | RBLNUNet2DConditionModelConfig | None = None
    movq: dict[str, Any] | RBLNVQModelConfig | None = None

    def __init__(
        self,
        *,
        sample_size: Optional[Tuple[int, int]] = None,
        batch_size: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        image_size: Optional[Tuple[int, int]] = None,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **data: Any,
    ):
        """
        Args:
            unet (Optional[RBLNUNet2DConditionModelConfig]): Configuration for the UNet model component.
                Initialized as RBLNUNet2DConditionModelConfig if not provided.
            movq (Optional[RBLNVQModelConfig]): Configuration for the MoVQ (VQ-GAN) model component.
                Initialized as RBLNVQModelConfig if not provided.
            sample_size (Optional[Tuple[int, int]]): Spatial dimensions for the UNet model.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            guidance_scale (Optional[float]): Scale for classifier-free guidance.
            image_size (Optional[Tuple[int, int]]): Dimensions for the generated images.
                Cannot be used together with img_height/img_width.
            img_height (Optional[int]): Height of the generated images.
            img_width (Optional[int]): Width of the generated images.
            height (Optional[int]): Height of the generated images.
            width (Optional[int]): Width of the generated images.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

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

        self.unet = self.initialize_submodule_config(
            self.unet,
            cls_name="RBLNUNet2DConditionModelConfig",
            sample_size=sample_size,
        )
        self.movq = self.initialize_submodule_config(
            self.movq,
            cls_name="RBLNVQModelConfig",
            batch_size=batch_size,
            sample_size=image_size,  # image size is equal to sample size in vae
            uses_encoder=self._movq_uses_encoder,
        )

        # Get default guidance scale from original class to set UNet batch size
        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]

        if not self.unet.batch_size_is_specified:
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.unet.batch_size = self.movq.batch_size * 2
            else:
                self.unet.batch_size = self.movq.batch_size

    @property
    def batch_size(self):
        return self.movq.batch_size

    @property
    def image_size(self):
        return self.movq.sample_size


class RBLNKandinskyV22PipelineConfig(RBLNKandinskyV22PipelineBaseConfig):
    """Configuration class for the Kandinsky V2.2 text-to-image decoder pipeline."""

    _movq_uses_encoder: ClassVar[bool] = False


class RBLNKandinskyV22Img2ImgPipelineConfig(RBLNKandinskyV22PipelineBaseConfig):
    """Configuration class for the Kandinsky V2.2 image-to-image decoder pipeline."""

    _movq_uses_encoder: ClassVar[bool] = True


class RBLNKandinskyV22InpaintPipelineConfig(RBLNKandinskyV22PipelineBaseConfig):
    """Configuration class for the Kandinsky V2.2 inpainting decoder pipeline."""

    _movq_uses_encoder: ClassVar[bool] = True


class RBLNKandinskyV22PriorPipelineConfig(RBLNModelConfig):
    """Configuration class for the Kandinsky V2.2 Prior pipeline."""

    submodules: ClassVar[list[str]] = ["text_encoder", "image_encoder", "prior"]

    text_encoder: dict[str, Any] | RBLNCLIPTextModelWithProjectionConfig | None = None
    image_encoder: dict[str, Any] | RBLNCLIPVisionModelWithProjectionConfig | None = None
    prior: dict[str, Any] | RBLNPriorTransformerConfig | None = None

    def __init__(
        self,
        *,
        batch_size: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        **data: Any,
    ):
        """
        Initialize a configuration for Kandinsky 2.2 prior pipeline optimized for RBLN NPU.

        This configuration sets up the prior components of the Kandinsky 2.2 architecture, which includes
        text and image encoders along with a prior transformer that maps text/image embeddings to
        latent representations used to condition the diffusion process.

        Args:
            text_encoder (Optional[RBLNCLIPTextModelWithProjectionConfig]): Configuration for the text encoder component.
                Initialized as RBLNCLIPTextModelWithProjectionConfig if not provided.
            image_encoder (Optional[RBLNCLIPVisionModelWithProjectionConfig]): Configuration for the image encoder component.
                Initialized as RBLNCLIPVisionModelWithProjectionConfig if not provided.
            prior (Optional[RBLNPriorTransformerConfig]): Configuration for the prior transformer component.
                Initialized as RBLNPriorTransformerConfig if not provided.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            guidance_scale (Optional[float]): Scale for classifier-free guidance.
            **data: Additional arguments passed to the parent RBLNModelConfig.

        Note:
            When guidance_scale > 1.0, the prior batch size is automatically doubled to
            accommodate classifier-free guidance.
        """
        super().__init__(**data)
        self.text_encoder = self.initialize_submodule_config(
            self.text_encoder,
            cls_name="RBLNCLIPTextModelWithProjectionConfig",
            batch_size=batch_size,
        )
        self.image_encoder = self.initialize_submodule_config(
            self.image_encoder,
            cls_name="RBLNCLIPVisionModelWithProjectionConfig",
            batch_size=batch_size,
        )
        self.prior = self.initialize_submodule_config(
            self.prior,
            cls_name="RBLNPriorTransformerConfig",
        )

        # Get default guidance scale from original class to set UNet batch size
        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]

        if not self.prior.batch_size_is_specified:
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.prior.batch_size = self.text_encoder.batch_size * 2
            else:
                self.prior.batch_size = self.text_encoder.batch_size

    @property
    def batch_size(self):
        return self.text_encoder.batch_size

    @property
    def image_size(self):
        return self.image_encoder.image_size


class RBLNKandinskyV22CombinedPipelineBaseConfig(RBLNModelConfig):
    """Base configuration class for Kandinsky V2.2 combined pipelines."""

    submodules: ClassVar[list[str]] = ["prior_pipe", "decoder_pipe"]
    _decoder_pipe_cls: ClassVar[type] = RBLNKandinskyV22PipelineConfig

    prior_pipe: dict[str, Any] | RBLNKandinskyV22PriorPipelineConfig | None = None
    # Use RBLNModelConfig to allow subclasses (Img2Img, Inpaint) to be assigned
    decoder_pipe: dict[str, Any] | RBLNModelConfig | None = None

    def __init__(
        self,
        *,
        sample_size: Optional[Tuple[int, int]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        batch_size: Optional[int] = None,
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        prior_prior: Optional[RBLNPriorTransformerConfig] = None,
        prior_image_encoder: Optional[RBLNCLIPVisionModelWithProjectionConfig] = None,
        prior_text_encoder: Optional[RBLNCLIPTextModelWithProjectionConfig] = None,
        unet: Optional[RBLNUNet2DConditionModelConfig] = None,
        movq: Optional[RBLNVQModelConfig] = None,
        **data: Any,
    ):
        """
        Initialize a configuration for combined Kandinsky 2.2 pipelines optimized for RBLN NPU.

        This configuration integrates both the prior and decoder components of Kandinsky 2.2 into
        a unified pipeline, allowing for end-to-end text-to-image generation in a single model.
        It combines the text/image encoding, prior mapping, and diffusion steps together.

        Args:
            prior_pipe (Optional[RBLNKandinskyV22PriorPipelineConfig]): Configuration for the prior pipeline.
                Initialized as RBLNKandinskyV22PriorPipelineConfig if not provided.
            decoder_pipe (Optional[RBLNKandinskyV22PipelineConfig]): Configuration for the decoder pipeline.
                Initialized as RBLNKandinskyV22PipelineConfig if not provided.
            sample_size (Optional[Tuple[int, int]]): Spatial dimensions for the UNet model.
            image_size (Optional[Tuple[int, int]]): Dimensions for the generated images.
                Cannot be used together with img_height/img_width.
            batch_size (Optional[int]): Batch size for inference, applied to all submodules.
            img_height (Optional[int]): Height of the generated images.
            img_width (Optional[int]): Width of the generated images.
            height (Optional[int]): Height of the generated images.
            width (Optional[int]): Width of the generated images.
            guidance_scale (Optional[float]): Scale for classifier-free guidance.
            prior_prior (Optional[RBLNPriorTransformerConfig]): Direct configuration for the prior transformer.
                Used if prior_pipe is not provided.
            prior_image_encoder (Optional[RBLNCLIPVisionModelWithProjectionConfig]): Direct configuration for the image encoder.
                Used if prior_pipe is not provided.
            prior_text_encoder (Optional[RBLNCLIPTextModelWithProjectionConfig]): Direct configuration for the text encoder.
                Used if prior_pipe is not provided.
            unet (Optional[RBLNUNet2DConditionModelConfig]): Direct configuration for the UNet.
                Used if decoder_pipe is not provided.
            movq (Optional[RBLNVQModelConfig]): Direct configuration for the MoVQ (VQ-GAN) model.
                Used if decoder_pipe is not provided.
            **data: Additional arguments passed to the parent RBLNModelConfig.
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

        self.prior_pipe = self.initialize_submodule_config(
            self.prior_pipe,
            cls_name="RBLNKandinskyV22PriorPipelineConfig",
            prior=prior_prior,
            image_encoder=prior_image_encoder,
            text_encoder=prior_text_encoder,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
        )
        self.decoder_pipe = self.initialize_submodule_config(
            self.decoder_pipe,
            cls_name=self._decoder_pipe_cls.__name__,
            unet=unet,
            movq=movq,
            batch_size=batch_size,
            sample_size=sample_size,
            image_size=image_size,
            guidance_scale=guidance_scale,
        )

    @property
    def batch_size(self):
        return self.prior_pipe.batch_size

    @property
    def image_size(self):
        return self.prior_pipe.image_size

    @property
    def prior_prior(self):
        return self.prior_pipe.prior

    @property
    def prior_image_encoder(self):
        return self.prior_pipe.image_encoder

    @property
    def prior_text_encoder(self):
        return self.prior_pipe.text_encoder

    @property
    def unet(self):
        return self.decoder_pipe.unet

    @property
    def movq(self):
        return self.decoder_pipe.movq


class RBLNKandinskyV22CombinedPipelineConfig(RBLNKandinskyV22CombinedPipelineBaseConfig):
    """Configuration class for the Kandinsky V2.2 combined text-to-image pipeline."""

    _decoder_pipe_cls: ClassVar[type] = RBLNKandinskyV22PipelineConfig


class RBLNKandinskyV22InpaintCombinedPipelineConfig(RBLNKandinskyV22CombinedPipelineBaseConfig):
    """Configuration class for the Kandinsky V2.2 combined inpainting pipeline."""

    _decoder_pipe_cls: ClassVar[type] = RBLNKandinskyV22InpaintPipelineConfig


class RBLNKandinskyV22Img2ImgCombinedPipelineConfig(RBLNKandinskyV22CombinedPipelineBaseConfig):
    """Configuration class for the Kandinsky V2.2 combined image-to-image pipeline."""

    _decoder_pipe_cls: ClassVar[type] = RBLNKandinskyV22Img2ImgPipelineConfig
