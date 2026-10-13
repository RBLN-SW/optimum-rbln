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

from diffusers import CogVideoXImageToVideoPipeline

from ...configurations import RBLNCogVideoXImageToVideoPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin
from .pipeline_cogvideox import RBLNCogVideoXPromptDtypeMixin


class RBLNCogVideoXImageToVideoPipeline(
    RBLNCogVideoXPromptDtypeMixin, RBLNDiffusionMixin, CogVideoXImageToVideoPipeline
):
    """
    RBLN-accelerated implementation of the CogVideoX image-to-video pipeline (e.g. CogVideoX-5b-I2V).

    The T5 text encoder, the 3D transformer and the VAE (encoder for the input image, decoder for the
    video) are compiled for RBLN NPUs. The pipeline builds the RoPE tables on the host and passes them to
    the transformer graph.
    """

    original_class = CogVideoXImageToVideoPipeline
    _rbln_config_class = RBLNCogVideoXImageToVideoPipelineConfig
    _submodules = ["text_encoder", "transformer", "vae"]


__all__ = [
    "RBLNCogVideoXImageToVideoPipeline",
]
