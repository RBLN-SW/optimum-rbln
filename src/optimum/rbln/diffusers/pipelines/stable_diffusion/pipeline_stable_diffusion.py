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


from diffusers import StableDiffusionPipeline

from ...configurations import RBLNStableDiffusionPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNStableDiffusionPipeline(RBLNDiffusionMixin, StableDiffusionPipeline):
    """
    RBLN-accelerated implementation of Stable Diffusion pipeline for text-to-image generation.

    This pipeline compiles Stable Diffusion models to run efficiently on RBLN NPUs, enabling high-performance
    inference for generating images from text prompts with optimized memory usage and throughput.
    """

    original_class = StableDiffusionPipeline
    _rbln_config_class = RBLNStableDiffusionPipelineConfig
    _submodules = ["vae", "text_encoder", "unet"]


__all__ = [
    "RBLNStableDiffusionPipeline",
]
