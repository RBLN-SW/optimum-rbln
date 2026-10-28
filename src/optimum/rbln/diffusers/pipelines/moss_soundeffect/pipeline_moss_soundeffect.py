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

from ...configurations import RBLNMossSoundEffectPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin
from .pipeline_moss_soundeffect_upstream import MossSoundEffectPipeline


class RBLNMossSoundEffectPipeline(RBLNDiffusionMixin, MossSoundEffectPipeline):
    """
    RBLN-accelerated implementation of MOSS-SoundEffect pipeline for text-to-audio generation.

    This pipeline compiles the Qwen3 text encoder, the WanAudio transformer and the DAC decoder for
    `max_inference_seconds` of audio and 512 prompt tokens to run efficiently on RBLN NPUs; a call denoises that
    length and crops the waveform to `seconds`.
    """

    original_class = MossSoundEffectPipeline
    _rbln_config_class = RBLNMossSoundEffectPipelineConfig
    _submodules = ["text_encoder", "transformer", "vae"]

    def handle_additional_kwargs(self, **kwargs):
        # If there is no max_inference_seconds of kwargs,
        # it is filled based on the compiled value.
        if kwargs.get("max_inference_seconds") is None:
            kwargs["max_inference_seconds"] = (
                self.vae.rbln_config.num_latent_frames * self.vae.hop_length // self.config.sample_rate
            )
        return kwargs


__all__ = [
    "RBLNMossSoundEffectPipeline",
]
