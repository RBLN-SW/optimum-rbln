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


from diffusers import QwenImageEditPipeline

from ...configurations import RBLNQwenImageEditPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNQwenImageEditPipeline(RBLNDiffusionMixin, QwenImageEditPipeline):
    """
    RBLN-accelerated implementation of QwenImageEditPipeline for image editing.

    This pipeline compiles Qwen-Image-Edit models to run efficiently on RBLN NPUs, enabling
    high-performance inference for editing images based on text prompts with semantic and
    appearance editing capabilities.
    """

    original_class = QwenImageEditPipeline
    _rbln_config_class = RBLNQwenImageEditPipelineConfig
    _submodules = ["text_encoder", "transformer", "vae"]
