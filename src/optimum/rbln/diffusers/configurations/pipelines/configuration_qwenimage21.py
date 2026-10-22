# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNQwen3VLForConditionalGenerationConfig
from ..models import RBLNAutoencoderKLQwenImage21Config, RBLNQwenImage21Transformer2DModelConfig
from ..models.configuration_autoencoder_kl_qwenimage21 import _image_sizes


# One latent token covers a 16x16 pixel tile. The vision encoder also reads 16x16 patches and merges 2x2 of them into
# one image token of the text encoder, so a condition image takes a quarter of its latent tokens in the prompt.
_LATENT_TILE = 16
_PATCHES_PER_IMAGE_TOKEN = 4
# The pipeline rounds image sizes down to multiples of 32.
_SIZE_MULTIPLE = 32
_SEQ_LEN_MULTIPLE = 64
# The pipeline resizes a condition image to `output_resolution**2` pixels at its aspect ratio, 1024**2 by default, so
# a square condition image becomes 1024x1024.
_DEFAULT_CONDITION_IMAGE_SIZE = (1024, 1024)


class RBLNQwenImage21PipelineConfig(RBLNModelConfig):
    """
    Configuration class for the RBLN Qwen-Image 2.1 pipeline.

    The pipeline compiles its text encoder, transformer and VAE for a fixed set of image sizes. Text-to-image needs
    only the output sizes. Image editing also needs the sizes of the condition images after the pipeline resizes
    them, for the vision encoder of the text encoder and for the VAE encoder. The pipeline compiles for image editing
    by default, and a call without condition images skips those parts.
    """

    submodules = ["text_encoder", "transformer", "vae"]

    def __init__(
        self,
        text_encoder: RBLNQwen3VLForConditionalGenerationConfig | None = None,
        transformer: RBLNQwenImage21Transformer2DModelConfig | None = None,
        vae: RBLNAutoencoderKLQwenImage21Config | None = None,
        *,
        batch_size: int | None = None,
        image_size: tuple[int, int] | list[tuple[int, int]] | None = None,
        condition_image_size: tuple[int, int] | list[tuple[int, int]] | None = None,
        num_condition_images: int | None = None,
        max_seq_len: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            text_encoder (RBLNQwen3VLForConditionalGenerationConfig | None): Configuration for the text encoder.
                Initialized as RBLNQwen3VLForConditionalGenerationConfig if not provided.
            transformer (RBLNQwenImage21Transformer2DModelConfig | None): Configuration for the transformer.
                Initialized as RBLNQwenImage21Transformer2DModelConfig if not provided.
            vae (RBLNAutoencoderKLQwenImage21Config | None): Configuration for the VAE.
                Initialized as RBLNAutoencoderKLQwenImage21Config if not provided.
            batch_size (int | None): Batch size for inference, applied to all submodules.
            image_size (tuple[int, int] | list[tuple[int, int]] | None): The (height, width) of the generated image,
                in pixels, or a list of them. Defaults to (1024, 1024).
            condition_image_size (tuple[int, int] | list[tuple[int, int]] | None): The (height, width) of the
                condition images for image editing, in pixels, or a list of them. These are the sizes after the
                pipeline resizes the images to about `output_resolution**2` pixels at their aspect ratio. Defaults to
                (1024, 1024), the size of a square condition image at the default `output_resolution`.
            num_condition_images (int | None): The maximum number of condition images in one call. Defaults to 1.
                Set it to 0 to compile for text-to-image only, which drops the vision and VAE encoders and the
                condition-image tokens from the sequence lengths and KV caches.
            max_seq_len (int | None): The maximum number of text tokens the text encoder reads, without the tokens
                of the condition images. Defaults to 512.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If an image size is not a multiple of 32, or if `condition_image_size` is set with
                `num_condition_images=0`.
        """
        super().__init__(**kwargs)

        image_size = _image_sizes(image_size) or [(1024, 1024)]
        condition_image_size = _image_sizes(condition_image_size)
        if num_condition_images is None:
            num_condition_images = 1
        if num_condition_images == 0 and condition_image_size:
            raise ValueError("condition_image_size needs num_condition_images of 1 or more.")
        # Empty when compiling for text-to-image only.
        condition_image_size = (
            (condition_image_size or [_DEFAULT_CONDITION_IMAGE_SIZE]) if num_condition_images else []
        )
        max_seq_len = max_seq_len or 512

        sizes = list(dict.fromkeys(image_size + condition_image_size))
        not_aligned = [size for size in sizes if size[0] % _SIZE_MULTIPLE or size[1] % _SIZE_MULTIPLE]
        if not_aligned:
            raise ValueError(
                f"Image sizes {not_aligned} are not multiples of {_SIZE_MULTIPLE}. The pipeline rounds the height and "
                f"width down to multiples of {_SIZE_MULTIPLE}, so it never runs these sizes."
            )

        def latent_tokens(size: tuple[int, int]) -> int:
            return (size[0] // _LATENT_TILE) * (size[1] // _LATENT_TILE)

        def round_up(length: int) -> int:
            # The compiler takes sequence lengths of the text and vision encoders only in multiples of 64.
            return math.ceil(length / _SEQ_LEN_MULTIPLE) * _SEQ_LEN_MULTIPLE

        condition_tokens = num_condition_images * max(map(latent_tokens, condition_image_size), default=0)
        visual: dict[str, Any]
        if condition_image_size:
            # The vision encoder runs once per condition image, on one patch per latent token, padded to the
            # smallest of its sequence lengths that holds them.
            visual = {"max_seq_len": sorted({round_up(latent_tokens(size)) for size in condition_image_size})}
        else:
            # Text-to-image never calls the vision encoder, so it compiles at its smallest size. Its runtime stays
            # on: `create_runtimes` would also apply when an image-editing artifact loads.
            visual = {"max_seq_len": 64}

        self.text_encoder = self.initialize_submodule_config(
            text_encoder,
            cls_name="RBLNQwen3VLForConditionalGenerationConfig",
            batch_size=batch_size,
            max_seq_len=round_up(max_seq_len + condition_tokens // _PATCHES_PER_IMAGE_TOKEN),
            phases=["prefill"],  # The pipeline reads the hidden states of one forward pass and never generates.
            output_hidden_states=True,
            visual=visual,
        )
        self.transformer = self.initialize_submodule_config(
            transformer,
            cls_name="RBLNQwenImage21Transformer2DModelConfig",
            batch_size=batch_size,
            sample_size=[(height // _LATENT_TILE, width // _LATENT_TILE) for height, width in sizes],
            # In the transformer every image token of the prompt becomes the latent tokens it covers.
            max_seq_len=max_seq_len + condition_tokens,
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLQwenImage21Config",
            batch_size=batch_size,
            sample_size=image_size,
            uses_encoder=bool(condition_image_size),
            encoder_sample_size=condition_image_size or None,
        )

    @property
    def batch_size(self):
        return self.vae.batch_size

    @property
    def image_size(self):
        return self.vae.sample_size


__all__ = [
    "RBLNQwenImage21PipelineConfig",
]
