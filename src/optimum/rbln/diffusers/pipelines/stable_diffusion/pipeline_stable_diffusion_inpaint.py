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

from diffusers import StableDiffusionInpaintPipeline

from ...configurations import RBLNStableDiffusionInpaintPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNStableDiffusionInpaintPipeline(RBLNDiffusionMixin, StableDiffusionInpaintPipeline):
    """
    RBLN-accelerated implementation of Stable Diffusion pipeline for image inpainting.

    This pipeline compiles Stable Diffusion models to run efficiently on RBLN NPUs, enabling high-performance
    inference for filling masked regions of images based on text prompts with seamless integration.
    """

    original_class = StableDiffusionInpaintPipeline
    _rbln_config_class = RBLNStableDiffusionInpaintPipelineConfig
    _submodules = ["text_encoder", "unet", "vae"]


__all__ = [
    "RBLNStableDiffusionInpaintPipeline",
]
