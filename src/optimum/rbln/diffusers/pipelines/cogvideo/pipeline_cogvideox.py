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

import torch
from diffusers import CogVideoXPipeline

from ...configurations import RBLNCogVideoXPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


class RBLNCogVideoXPromptDtypeMixin:
    def encode_prompt(self, *args, dtype: torch.dtype | None = None, **kwargs):
        # Without torch_dtype, transformers loads the T5 encoder at its checkpoint dtype (fp16 or bf16)
        # while diffusers loads the transformer at fp32, and each is compiled at that dtype. The pipeline
        # takes the latent dtype from the prompt embeddings, so return them in the transformer dtype to
        # keep the latents and the scheduler at the transformer dtype.
        return super().encode_prompt(*args, dtype=dtype or self.transformer.dtype, **kwargs)


class RBLNCogVideoXPipeline(RBLNCogVideoXPromptDtypeMixin, RBLNDiffusionMixin, CogVideoXPipeline):
    """
    RBLN-accelerated implementation of the CogVideoX text-to-video pipeline.

    The T5 text encoder, the 3D transformer and the VAE decoder are compiled for RBLN NPUs.
    The VAE decodes the latent video chunk by chunk with its temporal conv caches kept on the device.
    """

    original_class = CogVideoXPipeline
    _rbln_config_class = RBLNCogVideoXPipelineConfig
    _submodules = ["text_encoder", "transformer", "vae"]


__all__ = [
    "RBLNCogVideoXPipeline",
]
