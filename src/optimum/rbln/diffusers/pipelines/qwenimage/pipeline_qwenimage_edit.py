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


from diffusers import QwenImageEditPlusPipeline

from ...configurations import RBLNQwenImageEditPlusPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNQwenImageEditPlusPipeline(RBLNDiffusionMixin, QwenImageEditPlusPipeline):
    """
    RBLN-accelerated implementation of QwenImageEditPlusPipeline for image editing.

    This pipeline compiles Qwen-Image-Edit-Plus models to run efficiently on RBLN NPUs,
    enabling high-performance inference for editing images based on text prompts with
    semantic and appearance editing capabilities. Supports multiple input images and
    guidance-distilled models.
    """

    original_class = QwenImageEditPlusPipeline
    _rbln_config_class = RBLNQwenImageEditPlusPipelineConfig
    _submodules = ["text_encoder", "transformer", "vae"]
