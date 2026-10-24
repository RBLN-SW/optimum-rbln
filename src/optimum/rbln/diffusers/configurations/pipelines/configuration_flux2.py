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

from typing import Any

from ....configuration_utils import RBLNModelConfig
from ..models import RBLNAutoencoderKLFlux2Config, RBLNFlux2KleinTextEncoderConfig, RBLNFlux2Transformer2DModelConfig


class RBLNFlux2KleinPipelineConfig(RBLNModelConfig):
    """Static native geometry shared by the three FLUX.2 klein components."""

    submodules = ["text_encoder", "transformer", "vae"]

    def __init__(
        self,
        text_encoder: RBLNFlux2KleinTextEncoderConfig | dict | None = None,
        transformer: RBLNFlux2Transformer2DModelConfig | dict | None = None,
        vae: RBLNAutoencoderKLFlux2Config | dict | None = None,
        *,
        batch_size: int = 1,
        height: int = 1024,
        width: int = 1024,
        max_seq_len: int = 512,
        reference_image_count: int = 0,
        reference_image_size: tuple[int, int] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            text_encoder (RBLNFlux2KleinTextEncoderConfig | dict | None): Qwen3 encoding configuration.
            transformer (RBLNFlux2Transformer2DModelConfig | dict | None): Denoiser configuration.
            vae (RBLNAutoencoderKLFlux2Config | dict | None): Encoder/decoder configuration.
            batch_size (int): Compiled number of prompts and generated images.
            height (int): Output image height in pixels.
            width (int): Output image width in pixels.
            max_seq_len (int): Native prompt token length, at most 512.
            reference_image_count (int): Exact reference-image count per request.
            reference_image_size (tuple[int, int] | None): Reference geometry after native preprocessing.
            kwargs: Additional arguments passed to RBLNModelConfig.
        """
        super().__init__(**kwargs)
        self.text_encoder = self.initialize_submodule_config(
            text_encoder, cls_name="RBLNFlux2KleinTextEncoderConfig", batch_size=batch_size, max_seq_len=max_seq_len
        )
        self.transformer = self.initialize_submodule_config(
            transformer,
            cls_name="RBLNFlux2Transformer2DModelConfig",
            batch_size=batch_size,
            image_size=(height, width),
            prompt_embed_length=max_seq_len,
            reference_image_count=reference_image_count,
            reference_image_size=reference_image_size,
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLFlux2Config",
            batch_size=batch_size,
            sample_size=(height, width),
            reference_image_size=reference_image_size,
        )
        if (
            self.text_encoder.batch_size != self.transformer.batch_size
            or self.vae.batch_size != self.transformer.batch_size
        ):
            raise ValueError("All components must have the same compiled request batch")
        if self.text_encoder.max_seq_len != self.transformer.prompt_embed_length:
            raise ValueError("Text encoder and transformer must have the same prompt length")
        if self.vae.sample_size != self.transformer.image_size:
            raise ValueError("VAE and transformer must have the same output image geometry")
        if self.vae.reference_image_size != self.transformer.reference_image_size:
            raise ValueError("VAE and transformer must have the same reference image geometry")


__all__ = ["RBLNFlux2KleinPipelineConfig"]
