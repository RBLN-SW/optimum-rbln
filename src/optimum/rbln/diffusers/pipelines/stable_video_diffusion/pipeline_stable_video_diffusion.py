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


from diffusers import StableVideoDiffusionPipeline

from ....utils.logging import get_logger
from ...configurations import RBLNStableVideoDiffusionPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


logger = get_logger(__name__)


class RBLNStableVideoDiffusionPipeline(RBLNDiffusionMixin, StableVideoDiffusionPipeline):
    """
    RBLN-accelerated implementation of Stable Video Diffusion pipeline for image-to-video generation.

    This pipeline compiles Stable Video Diffusion models to run efficiently on RBLN NPUs, enabling high-performance
    inference for generating videos from images with optimized memory usage and throughput.
    """

    original_class = StableVideoDiffusionPipeline
    _rbln_config_class = RBLNStableVideoDiffusionPipelineConfig
    _submodules = ["image_encoder", "unet", "vae"]

    def handle_additional_kwargs(self, **kwargs):
        compiled_num_frames = self.unet.rbln_config.num_frames
        if compiled_num_frames is not None:
            kwargs["num_frames"] = compiled_num_frames

        compiled_decode_chunk_size = self.vae.rbln_config.decode_chunk_size
        if compiled_decode_chunk_size is not None:
            kwargs["decode_chunk_size"] = compiled_decode_chunk_size
        return kwargs
