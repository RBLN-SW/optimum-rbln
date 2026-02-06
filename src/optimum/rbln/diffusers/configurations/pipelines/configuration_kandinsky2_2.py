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
from ....transformers import RBLNCLIPTextModelWithProjectionConfig, RBLNCLIPVisionModelWithProjectionConfig
from ..models import RBLNUNet2DConditionModelConfig, RBLNVQModelConfig
from ..models.configuration_prior_transformer import RBLNPriorTransformerConfig


class RBLNKandinskyV22PipelineBaseConfig(RBLNModelConfig):
    """Base configuration for Kandinsky V2.2 pipelines."""

    submodules: ClassVar[list[str]] = ["unet", "movq"]
    _movq_uses_encoder: ClassVar[bool] = False

    unet: dict[str, Any] | RBLNUNet2DConditionModelConfig | None = Field(
        default=None, description="Configuration for the UNet model component."
    )
    movq: dict[str, Any] | RBLNVQModelConfig | None = Field(
        default=None, description="Configuration for the MoVQ (VQ-GAN) model component."
    )

    # Pass-through parameters (excluded from serialization.)
    effective_batch_size: int | None = Field(
        default=None,
        alias="batch_size",
        exclude=True,
        description="Batch size for inference. Forwarded to movq.",
    )
    effective_img_height: int | None = Field(
        default=None,
        alias="img_height",
        exclude=True,
        description="(Deprecated) Image height. Use height instead.",
    )
    effective_img_width: int | None = Field(
        default=None,
        alias="img_width",
        exclude=True,
        description="(Deprecated) Image width. Use width instead.",
    )
    effective_height: int | None = Field(
        default=None,
        alias="height",
        exclude=True,
        description="Height of the generated images.",
    )
    effective_width: int | None = Field(
        default=None,
        alias="width",
        exclude=True,
        description="Width of the generated images.",
    )
    effective_sample_size: tuple[int, int] | None = Field(
        default=None,
        alias="sample_size",
        exclude=True,
        description="Spatial dimensions for the UNet model (height, width).",
    )
    effective_image_size: tuple[int, int] | None = Field(
        default=None,
        alias="image_size",
        exclude=True,
        description="Image dimensions (height, width). Forwarded to movq as sample_size.",
    )
    effective_guidance_scale: float | None = Field(
        default=None,
        alias="guidance_scale",
        exclude=True,
        description="Scale for classifier-free guidance. Used to determine UNet batch size.",
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_image_dimensions(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Resolve image_size from height/width aliases."""
        if not isinstance(data, dict):
            return data

        # Check only user-facing keys (aliases), not internal effective_* keys
        image_size = data.get("image_size")
        img_height = data.get("img_height")
        img_width = data.get("img_width")
        height = data.get("height")
        width = data.get("width")

        # Initial check for image_size conflict
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
            data["image_size"] = (height, width)
        elif (height is not None and width is None) or (height is None and width is not None):
            raise ValueError("Both height and width must be provided together if used")
        # Fallback to img_height/img_width for backward compatibility
        elif img_height is not None and img_width is not None:
            data["image_size"] = (img_height, img_width)
        elif (img_height is not None and img_width is None) or (img_height is None and img_width is not None):
            raise ValueError("Both img_height and img_width must be provided together if used")

        return data

    @model_validator(mode="after")
    def initialize_submodules(self) -> "RBLNKandinskyV22PipelineBaseConfig":
        """Initialize submodule configs with pass-through parameters."""
        # Guard against re-entry during submodule initialization
        if getattr(self, "_submodules_initialized", False):
            return self
        object.__setattr__(self, "_submodules_initialized", True)

        self.unet = self.initialize_submodule_config(
            self.unet,
            cls_name="RBLNUNet2DConditionModelConfig",
            sample_size=self.effective_sample_size,
        )
        self.movq = self.initialize_submodule_config(
            self.movq,
            cls_name="RBLNVQModelConfig",
            batch_size=self.effective_batch_size,
            sample_size=self.effective_image_size,  # image size is equal to sample size in vae
            uses_encoder=self._movq_uses_encoder,
        )

        # Get default guidance scale from original class to set UNet batch size
        guidance_scale = self.effective_guidance_scale
        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]

        if not self.unet.batch_size_is_specified:
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.unet.batch_size = self.movq.batch_size * 2
            else:
                self.unet.batch_size = self.movq.batch_size

        return self

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

    text_encoder: dict[str, Any] | RBLNCLIPTextModelWithProjectionConfig | None = Field(
        default=None, description="Configuration for the text encoder component."
    )
    image_encoder: dict[str, Any] | RBLNCLIPVisionModelWithProjectionConfig | None = Field(
        default=None, description="Configuration for the image encoder component."
    )
    prior: dict[str, Any] | RBLNPriorTransformerConfig | None = Field(
        default=None, description="Configuration for the prior transformer component."
    )

    # Pass-through parameters (excluded from serialization.)
    effective_batch_size: int | None = Field(
        default=None,
        alias="batch_size",
        exclude=True,
        description="Batch size for inference. Forwarded to text_encoder and image_encoder.",
    )
    effective_guidance_scale: float | None = Field(
        default=None,
        alias="guidance_scale",
        exclude=True,
        description="Scale for classifier-free guidance. Used to determine prior batch size.",
    )

    @model_validator(mode="after")
    def initialize_submodules(self) -> "RBLNKandinskyV22PriorPipelineConfig":
        """Initialize submodule configs with pass-through parameters."""
        # Guard against re-entry during submodule initialization
        if getattr(self, "_submodules_initialized", False):
            return self
        object.__setattr__(self, "_submodules_initialized", True)

        self.text_encoder = self.initialize_submodule_config(
            self.text_encoder,
            cls_name="RBLNCLIPTextModelWithProjectionConfig",
            batch_size=self.effective_batch_size,
        )
        self.image_encoder = self.initialize_submodule_config(
            self.image_encoder,
            cls_name="RBLNCLIPVisionModelWithProjectionConfig",
            batch_size=self.effective_batch_size,
        )
        self.prior = self.initialize_submodule_config(
            self.prior,
            cls_name="RBLNPriorTransformerConfig",
        )

        # Get default guidance scale from original class to set prior batch size
        guidance_scale = self.effective_guidance_scale
        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]

        if not self.prior.batch_size_is_specified:
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.prior.batch_size = self.text_encoder.batch_size * 2
            else:
                self.prior.batch_size = self.text_encoder.batch_size

        return self

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

    prior_pipe: dict[str, Any] | RBLNKandinskyV22PriorPipelineConfig | None = Field(
        default=None, description="Configuration for the prior pipeline."
    )
    decoder_pipe: dict[str, Any] | RBLNModelConfig | None = Field(
        default=None, description="Configuration for the decoder pipeline."
    )

    # Pass-through parameters (excluded from serialization.)
    effective_batch_size: int | None = Field(
        default=None,
        alias="batch_size",
        exclude=True,
        description="Batch size for inference. Forwarded to prior_pipe and decoder_pipe.",
    )
    effective_img_height: int | None = Field(
        default=None,
        alias="img_height",
        exclude=True,
        description="(Deprecated) Image height. Use height instead.",
    )
    effective_img_width: int | None = Field(
        default=None,
        alias="img_width",
        exclude=True,
        description="(Deprecated) Image width. Use width instead.",
    )
    effective_height: int | None = Field(
        default=None,
        alias="height",
        exclude=True,
        description="Height of the generated images.",
    )
    effective_width: int | None = Field(
        default=None,
        alias="width",
        exclude=True,
        description="Width of the generated images.",
    )
    effective_sample_size: tuple[int, int] | None = Field(
        default=None,
        alias="sample_size",
        exclude=True,
        description="Spatial dimensions for the UNet model (height, width).",
    )
    effective_image_size: tuple[int, int] | None = Field(
        default=None,
        alias="image_size",
        exclude=True,
        description="Image dimensions (height, width). Forwarded to decoder_pipe.",
    )
    effective_guidance_scale: float | None = Field(
        default=None,
        alias="guidance_scale",
        exclude=True,
        description="Scale for classifier-free guidance.",
    )
    effective_prior_prior: RBLNPriorTransformerConfig | None = Field(
        default=None,
        alias="prior_prior",
        exclude=True,
        description="Configuration for the prior transformer in the prior pipeline.",
    )
    effective_prior_image_encoder: RBLNCLIPVisionModelWithProjectionConfig | None = Field(
        default=None,
        alias="prior_image_encoder",
        exclude=True,
        description="Configuration for the image encoder in the prior pipeline.",
    )
    effective_prior_text_encoder: RBLNCLIPTextModelWithProjectionConfig | None = Field(
        default=None,
        alias="prior_text_encoder",
        exclude=True,
        description="Configuration for the text encoder in the prior pipeline.",
    )
    effective_unet: RBLNUNet2DConditionModelConfig | None = Field(
        default=None,
        alias="unet",
        exclude=True,
        description="Configuration for the UNet in the decoder pipeline.",
    )
    effective_movq: RBLNVQModelConfig | None = Field(
        default=None,
        alias="movq",
        exclude=True,
        description="Configuration for the MoVQ in the decoder pipeline.",
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_image_dimensions(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Resolve image_size from height/width aliases."""
        if not isinstance(data, dict):
            return data

        # Check only user-facing keys (aliases), not internal effective_* keys
        image_size = data.get("image_size")
        img_height = data.get("img_height")
        img_width = data.get("img_width")
        height = data.get("height")
        width = data.get("width")

        # Initial check for image_size conflict
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
            data["image_size"] = (height, width)
        elif (height is not None and width is None) or (height is None and width is not None):
            raise ValueError("Both height and width must be provided together if used")
        # Fallback to img_height/img_width for backward compatibility
        elif img_height is not None and img_width is not None:
            data["image_size"] = (img_height, img_width)
        elif (img_height is not None and img_width is None) or (img_height is None and img_width is not None):
            raise ValueError("Both img_height and img_width must be provided together if used")

        return data

    @model_validator(mode="after")
    def initialize_submodules(self) -> "RBLNKandinskyV22CombinedPipelineBaseConfig":
        """Initialize submodule configs with pass-through parameters."""
        # Guard against re-entry during submodule initialization
        if getattr(self, "_submodules_initialized", False):
            return self
        object.__setattr__(self, "_submodules_initialized", True)

        self.prior_pipe = self.initialize_submodule_config(
            self.prior_pipe,
            cls_name="RBLNKandinskyV22PriorPipelineConfig",
            prior=self.effective_prior_prior,
            image_encoder=self.effective_prior_image_encoder,
            text_encoder=self.effective_prior_text_encoder,
            batch_size=self.effective_batch_size,
            guidance_scale=self.effective_guidance_scale,
        )
        self.decoder_pipe = self.initialize_submodule_config(
            self.decoder_pipe,
            cls_name=self._decoder_pipe_cls.__name__,
            unet=self.effective_unet,
            movq=self.effective_movq,
            batch_size=self.effective_batch_size,
            sample_size=self.effective_sample_size,
            image_size=self.effective_image_size,
            guidance_scale=self.effective_guidance_scale,
        )

        return self

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
