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

import inspect

from diffusers import Flux2KleinPipeline

from ...configurations.pipelines.configuration_flux2 import RBLNFlux2KleinPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNFlux2KleinPipeline(RBLNDiffusionMixin, Flux2KleinPipeline):
    """FLUX.2 klein using the original tokenizer, scheduler and denoising pipeline.

    The compiled profile fixes prompt length, request batch, output geometry and
    reference count/geometry. Reference images retain native preprocessing and
    token concatenation; no padding tokens are added to attention.
    """

    original_class = Flux2KleinPipeline
    _rbln_config_class = RBLNFlux2KleinPipelineConfig
    _submodules = ["text_encoder", "transformer", "vae"]
    _native_call_signature = inspect.signature(Flux2KleinPipeline.__call__)

    def __call__(self, *args, **kwargs):
        arguments = RBLNFlux2KleinPipeline._native_call_signature.bind(self, *args, **kwargs).arguments
        arguments.pop("self")
        config = self.transformer.rbln_config
        height, width = config.image_size
        if arguments.get("height") not in (None, height) or arguments.get("width") not in (None, width):
            raise ValueError("Requested image geometry does not match the compiled profile")
        arguments["height"], arguments["width"] = height, width
        length = arguments.get("max_sequence_length", config.prompt_embed_length)
        if length != config.prompt_embed_length:
            raise ValueError("Requested prompt length does not match the compiled profile")
        arguments["max_sequence_length"] = length
        prompt = arguments.get("prompt")
        embeds = arguments.get("prompt_embeds")
        prompts = embeds.shape[0] if embeds is not None else len(prompt) if isinstance(prompt, list) else 1
        if prompts != self.text_encoder.rbln_config.batch_size:
            raise ValueError("Prompt batch does not match the compiled text encoder")
        if prompts * arguments.get("num_images_per_prompt", 1) != config.batch_size:
            raise ValueError("Generated image batch does not match the compiled profile")
        image = arguments.get("image")
        references = 0 if image is None else len(image) if isinstance(image, list) else 1
        if references != config.reference_image_count:
            raise ValueError("Reference image count does not match the compiled profile")
        return Flux2KleinPipeline.__call__(self, **arguments)

    def prepare_image_latents(self, images, batch_size, generator, device, dtype):
        config = self.transformer.rbln_config
        if len(images) != config.reference_image_count or any(
            tuple(image.shape) != (1, 3, *config.reference_image_size) for image in images
        ):
            raise ValueError("Native preprocessed reference geometry does not match the compiled profile")
        return super().prepare_image_latents(images, batch_size, generator, device, dtype)


__all__ = ["RBLNFlux2KleinPipeline"]
