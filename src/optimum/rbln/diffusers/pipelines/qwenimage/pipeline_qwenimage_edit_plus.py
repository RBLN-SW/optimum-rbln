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

import logging
import math

from diffusers import QwenImageEditPlusPipeline

from ...configurations import RBLNQwenImageEditPlusPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


logger = logging.getLogger(__name__)

VAE_IMAGE_SIZE = 1024 * 1024


def _calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


class RBLNQwenImageEditPlusPipeline(RBLNDiffusionMixin, QwenImageEditPlusPipeline):
    """
    RBLN-accelerated implementation of QwenImageEditPlusPipeline for image editing.

    This pipeline compiles Qwen-Image-Edit-Plus models to run efficiently on RBLN NPUs,
    enabling high-performance inference for editing images based on text prompts with
    semantic and appearance editing capabilities. Currently supports single-image
    input only; multi-image editing is not yet supported due to compile-time
    constraints on img_shapes.
    """

    original_class = QwenImageEditPlusPipeline
    _rbln_config_class = RBLNQwenImageEditPlusPipelineConfig
    _submodules = ["text_encoder", "transformer", "vae"]

    def handle_additional_kwargs(self, **kwargs):
        compiled_image_size = self.vae.rbln_config.image_size
        if compiled_image_size is not None:
            compiled_h, compiled_w = compiled_image_size

            user_h = kwargs.get("height")
            user_w = kwargs.get("width")
            if user_h is not None and user_w is not None:
                if user_h != compiled_h or user_w != compiled_w:
                    raise ValueError(
                        f"Requested output size ({user_h}x{user_w}) differs from compiled "
                        f"image_size ({compiled_h}x{compiled_w}). "
                        f"Either omit height/width to use the compiled size, "
                        f"or recompile the pipeline with the desired dimensions."
                    )

        compiled_prompt_len = getattr(self.transformer, "rbln_config", None)
        if compiled_prompt_len is not None:
            compiled_prompt_len = getattr(compiled_prompt_len, "prompt_embed_length", None)
        if compiled_prompt_len is not None:
            user_val = kwargs.get("max_sequence_length")
            if user_val is not None and user_val != compiled_prompt_len:
                logger.warning(
                    f"max_sequence_length={user_val} differs from compiled "
                    f"prompt_embed_length={compiled_prompt_len}. Overriding to compiled value."
                )
            kwargs["max_sequence_length"] = compiled_prompt_len

        image = kwargs.get("image")
        if image is not None:
            images = image if isinstance(image, list) else [image]
            if len(images) > 1:
                raise NotImplementedError(
                    f"RBLNQwenImageEditPlusPipeline currently supports only single-image input, "
                    f"but received {len(images)} images. Multi-image editing requires dynamic "
                    f"img_shapes which cannot be fixed at compile time."
                )

        if image is not None and compiled_image_size is not None:
            compiled_h, compiled_w = compiled_image_size
            images = image if isinstance(image, list) else [image]
            for idx, img in enumerate(images):
                if hasattr(img, "size"):
                    img_w, img_h = img.size
                    vae_w, vae_h = _calculate_dimensions(VAE_IMAGE_SIZE, img_w / img_h)

                    multiple_of = self.vae_scale_factor * 2
                    vae_w = vae_w // multiple_of * multiple_of
                    vae_h = vae_h // multiple_of * multiple_of

                    if vae_h != compiled_h or vae_w != compiled_w:
                        raise ValueError(
                            f"Input image[{idx}] has size ({img_w}x{img_h}), which after "
                            f"resizing for VAE becomes ({vae_w}x{vae_h}). This does not match "
                            f"the compiled image_size ({compiled_h}x{compiled_w}). "
                            f"Please provide images whose aspect ratio results in "
                            f"({compiled_h}x{compiled_w}) after VAE preprocessing, "
                            f"or recompile the pipeline with a matching image_size."
                        )

        return kwargs
