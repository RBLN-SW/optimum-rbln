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
from ....transformers import RBLNQwen3_5ModelConfig
from ..models import RBLNAutoencoderKLFlux2Config, RBLNMOSSImageTransformerConfig


# One latent token covers a 16x16 pixel tile: the VAE compresses 8x and the pipeline patchifies 2x2.
_LATENT_TILE = 16
# The compiler takes the sequence lengths of the text encoder and its vision encoder in multiples of 64.
_SEQ_LEN_MULTIPLE = 64


class RBLNMOSSImagePipelineConfig(RBLNModelConfig):
    """
    Configuration class for the RBLN MOSS-Image pipeline.

    The pipeline compiles its text encoder, transformer and VAE decoder for one image size and one prompt length.
    """

    submodules = ["text_encoder", "transformer", "vae"]

    def __init__(
        self,
        text_encoder: RBLNQwen3_5ModelConfig | None = None,
        transformer: RBLNMOSSImageTransformerConfig | None = None,
        vae: RBLNAutoencoderKLFlux2Config | None = None,
        *,
        batch_size: int | None = None,
        height: int | None = None,
        width: int | None = None,
        max_sequence_length: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            text_encoder (RBLNQwen3_5ModelConfig | None): Configuration for the Qwen3.5 text encoder. It encodes one
                prompt at a time, as upstream does, so its batch size is 1. Its vision encoder never runs: it compiles
                at the smallest size and gets no runtime.
            transformer (RBLNMOSSImageTransformerConfig | None): Configuration for the transformer.
            vae (RBLNAutoencoderKLFlux2Config | None): Configuration for the VAE, which only decodes.
            batch_size (int | None): The number of images generated per call (prompts times `num_images_per_prompt`),
                for the transformer and the VAE. Defaults to 1.
            height (int | None): The height of the generated image, a multiple of 16. Defaults to 1024.
            width (int | None): The width of the generated image, a multiple of 16. Defaults to 1024.
            max_sequence_length (int | None): The number of text tokens the text encoder and the transformer read.
                A longer prompt is truncated to it, as `max_sequence_length` of the pipeline call truncates upstream.
                Defaults to 512.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If only one of height and width is given, or if either is not a multiple of 16.
        """
        super().__init__(**kwargs)
        if (height is None) != (width is None):
            raise ValueError("Both height and width must be provided together if used")

        defaults = self.get_default_values_for_original_cls("__call__", ["height", "width"])
        height = height or defaults["height"]
        width = width or defaults["width"]
        if height % _LATENT_TILE or width % _LATENT_TILE:
            raise ValueError(f"height and width must be multiples of {_LATENT_TILE}, got {height}x{width}.")
        max_sequence_length = max_sequence_length or 512

        self.text_encoder = self.initialize_submodule_config(
            text_encoder,
            cls_name="RBLNQwen3_5ModelConfig",
            batch_size=1,
            use_inputs_embeds=True,
            max_seq_len=math.ceil(max_sequence_length / _SEQ_LEN_MULTIPLE) * _SEQ_LEN_MULTIPLE,
            visual={"max_seq_len": _SEQ_LEN_MULTIPLE, "create_runtimes": False},
        )
        self.transformer = self.initialize_submodule_config(
            transformer,
            cls_name="RBLNMOSSImageTransformerConfig",
            batch_size=batch_size,
            sample_size=(height // _LATENT_TILE, width // _LATENT_TILE),
            max_sequence_length=max_sequence_length,
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLFlux2Config",
            batch_size=batch_size,
            sample_size=(height, width),
            uses_encoder=False,
        )

    @property
    def batch_size(self):
        return self.transformer.batch_size

    @property
    def image_size(self):
        return self.vae.sample_size

    @property
    def max_sequence_length(self):
        return self.transformer.max_sequence_length


__all__ = [
    "RBLNMOSSImagePipelineConfig",
]
