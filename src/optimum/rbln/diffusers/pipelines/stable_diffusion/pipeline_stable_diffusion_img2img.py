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

from diffusers import StableDiffusionImg2ImgPipeline

from ...configurations import RBLNStableDiffusionImg2ImgPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNStableDiffusionImg2ImgPipeline(RBLNDiffusionMixin, StableDiffusionImg2ImgPipeline):
    """
    RBLN-accelerated implementation of Stable Diffusion pipeline for image-to-image generation.

    This pipeline compiles Stable Diffusion models to run efficiently on RBLN NPUs, enabling high-performance
    inference for transforming input images based on text prompts with controlled strength and guidance.
    """

    original_class = StableDiffusionImg2ImgPipeline
    _rbln_config_class = RBLNStableDiffusionImg2ImgPipelineConfig
    _submodules = ["text_encoder", "unet", "vae"]


__all__ = [
    "RBLNStableDiffusionImg2ImgPipeline",
]
