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
from ....transformers import RBLNCLIPTextModelWithProjectionConfig, RBLNT5EncoderModelConfig
from ..models import RBLNAutoencoderKLConfig, RBLNSD3Transformer2DModelConfig


class RBLNStableDiffusion3PipelineBaseConfig(RBLNModelConfig):
    """Base configuration for Stable Diffusion 3 pipelines."""

    submodules: ClassVar[list[str]] = ["transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "vae"]
    _vae_uses_encoder: ClassVar[bool] = False

    transformer: dict[str, Any] | RBLNSD3Transformer2DModelConfig | None = Field(
        default=None, description="Configuration for the transformer model component."
    )
    text_encoder: dict[str, Any] | RBLNCLIPTextModelWithProjectionConfig | None = Field(
        default=None, description="Configuration for the primary text encoder."
    )
    text_encoder_2: dict[str, Any] | RBLNCLIPTextModelWithProjectionConfig | None = Field(
        default=None, description="Configuration for the secondary text encoder."
    )
    text_encoder_3: dict[str, Any] | RBLNT5EncoderModelConfig | None = Field(
        default=None, description="Configuration for the tertiary text encoder (T5)."
    )
    vae: dict[str, Any] | RBLNAutoencoderKLConfig | None = Field(
        default=None, description="Configuration for the VAE model component."
    )

    # Pass-through parameters (excluded from serialization.)
    effective_batch_size: int | None = Field(
        default=None,
        alias="batch_size",
        exclude=True,
        description="Batch size for inference. Forwarded to text encoders and vae.",
    )
    effective_max_seq_len: int | None = Field(
        default=None,
        alias="max_seq_len",
        exclude=True,
        description="Maximum sequence length for text encoder 3 (T5). Defaults to 256.",
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
        description="Spatial dimensions for the transformer model (height, width).",
    )
    effective_image_size: tuple[int, int] | None = Field(
        default=None,
        alias="image_size",
        exclude=True,
        description="Image dimensions (height, width). Forwarded to vae as sample_size.",
    )
    effective_guidance_scale: float | None = Field(
        default=None,
        alias="guidance_scale",
        exclude=True,
        description="Scale for classifier-free guidance. Used to determine transformer batch size.",
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
    def initialize_submodules(self) -> "RBLNStableDiffusion3PipelineBaseConfig":
        """Initialize submodule configs with pass-through parameters."""
        # Guard against re-entry during submodule initialization
        if getattr(self, "_submodules_initialized", False):
            return self
        object.__setattr__(self, "_submodules_initialized", True)

        max_seq_len = self.effective_max_seq_len or 256

        self.text_encoder = self.initialize_submodule_config(
            self.text_encoder,
            cls_name="RBLNCLIPTextModelWithProjectionConfig",
            batch_size=self.effective_batch_size,
        )
        self.text_encoder_2 = self.initialize_submodule_config(
            self.text_encoder_2,
            cls_name="RBLNCLIPTextModelWithProjectionConfig",
            batch_size=self.effective_batch_size,
        )
        self.text_encoder_3 = self.initialize_submodule_config(
            self.text_encoder_3,
            cls_name="RBLNT5EncoderModelConfig",
            batch_size=self.effective_batch_size,
            max_seq_len=max_seq_len,
            model_input_names=["input_ids"],
        )
        self.transformer = self.initialize_submodule_config(
            self.transformer,
            cls_name="RBLNSD3Transformer2DModelConfig",
            sample_size=self.effective_sample_size,
        )
        self.vae = self.initialize_submodule_config(
            self.vae,
            cls_name="RBLNAutoencoderKLConfig",
            batch_size=self.effective_batch_size,
            uses_encoder=self.__class__._vae_uses_encoder,
            sample_size=self.effective_image_size,
        )

        # Get default guidance scale from original class to set Transformer batch size
        guidance_scale = self.effective_guidance_scale
        if guidance_scale is None:
            guidance_scale = self.get_default_values_for_original_cls("__call__", ["guidance_scale"])["guidance_scale"]

        if not self.transformer.batch_size_is_specified:
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                self.transformer.batch_size = self.text_encoder.batch_size * 2
            else:
                self.transformer.batch_size = self.text_encoder.batch_size

        return self

    @property
    def max_seq_len(self):
        return self.text_encoder_3.max_seq_len

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def sample_size(self):
        return self.transformer.sample_size

    @property
    def image_size(self):
        return self.vae.sample_size


class RBLNStableDiffusion3PipelineConfig(RBLNStableDiffusion3PipelineBaseConfig):
    """Config for SD3 Text2Img Pipeline"""

    _vae_uses_encoder: ClassVar[bool] = False


class RBLNStableDiffusion3Img2ImgPipelineConfig(RBLNStableDiffusion3PipelineBaseConfig):
    """Config for SD3 Img2Img Pipeline"""

    _vae_uses_encoder: ClassVar[bool] = True


class RBLNStableDiffusion3InpaintPipelineConfig(RBLNStableDiffusion3PipelineBaseConfig):
    """Config for SD3 Inpainting Pipeline"""

    _vae_uses_encoder: ClassVar[bool] = True
