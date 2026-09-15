# Copyright 2024 Rebellions Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from diffusers import KandinskyV22Pipeline

from ...configurations import RBLNKandinskyV22PipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNKandinskyV22Pipeline(RBLNDiffusionMixin, KandinskyV22Pipeline):
    """
    RBLN-accelerated implementation of Kandinsky 2.2 pipeline for text-to-image generation.

    This pipeline compiles Kandinsky 2.2 models to run efficiently on RBLN NPUs, enabling high-performance
    inference for generating images with distinctive artistic style and enhanced visual quality.
    """

    original_class = KandinskyV22Pipeline
    _rbln_config_class = RBLNKandinskyV22PipelineConfig
    _submodules = ["unet", "movq"]

    def get_compiled_image_size(self):
        return self.movq.image_size


__all__ = [
    "RBLNKandinskyV22Pipeline",
]
