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

from diffusers import KandinskyV22PriorPipeline

from ...configurations import RBLNKandinskyV22PriorPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNKandinskyV22PriorPipeline(RBLNDiffusionMixin, KandinskyV22PriorPipeline):
    """
    RBLN-accelerated implementation of Kandinsky 2.2 prior pipeline for text and image embedding generation.

    This pipeline compiles Kandinsky 2.2 prior models to run efficiently on RBLN NPUs, enabling high-performance
    inference for generating image embeddings from text prompts and image inputs for downstream generation tasks.
    """

    original_class = KandinskyV22PriorPipeline
    _rbln_config_class = RBLNKandinskyV22PriorPipelineConfig
    _submodules = ["text_encoder", "image_encoder", "prior"]


__all__ = [
    "RBLNKandinskyV22PriorPipeline",
]
