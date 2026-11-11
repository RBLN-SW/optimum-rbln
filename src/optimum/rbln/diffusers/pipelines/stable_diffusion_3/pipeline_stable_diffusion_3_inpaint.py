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

from diffusers import StableDiffusion3InpaintPipeline

from ...configurations import RBLNStableDiffusion3InpaintPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNStableDiffusion3InpaintPipeline(RBLNDiffusionMixin, StableDiffusion3InpaintPipeline):
    """
    RBLN-accelerated implementation of Stable Diffusion 3 pipeline for advanced image inpainting.

    This pipeline compiles Stable Diffusion 3 models to run efficiently on RBLN NPUs, enabling high-performance
    inference for filling masked regions with superior text understanding and seamless content generation.
    """

    original_class = StableDiffusion3InpaintPipeline
    _rbln_config_class = RBLNStableDiffusion3InpaintPipelineConfig
    _submodules = ["transformer", "text_encoder_3", "text_encoder", "text_encoder_2", "vae"]


__all__ = [
    "RBLNStableDiffusion3InpaintPipeline",
]
