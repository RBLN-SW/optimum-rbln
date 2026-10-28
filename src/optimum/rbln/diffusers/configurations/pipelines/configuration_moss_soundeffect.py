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

from typing import Any

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNQwen3ForCausalLMConfig
from ..models import RBLNDACConfig, RBLNWanAudioModelConfig


class RBLNMossSoundEffectPipelineConfig(RBLNModelConfig):
    """
    Configuration class for the RBLN MOSS-SoundEffect pipeline.

    The pipeline always denoises `max_inference_seconds` of audio, so the text encoder, the transformer and the VAE
    decoder compile for that one latent length and for 512 prompt tokens.
    """

    submodules = ["text_encoder", "transformer", "vae"]

    def __init__(
        self,
        text_encoder: RBLNQwen3ForCausalLMConfig | None = None,
        transformer: RBLNWanAudioModelConfig | None = None,
        vae: RBLNDACConfig | None = None,
        *,
        batch_size: int | None = None,
        max_inference_seconds: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            text_encoder (RBLNQwen3ForCausalLMConfig | None): Configuration for the Qwen3 text encoder. The pipeline
                encodes one prompt at a time, so its batch size is 1; it returns the hidden states of every layer, as
                upstream reads the last one.
            transformer (RBLNWanAudioModelConfig | None): Configuration for the transformer.
            vae (RBLNDACConfig | None): Configuration for the DAC VAE, which only decodes.
            batch_size (int | None): The number of prompts per call. Defaults to 1.
            max_inference_seconds (int | None): The seconds of audio the pipeline denoises per call (the waveform is
                cropped to `seconds` afterwards). Defaults to the checkpoint's `max_inference_seconds` (30).
            kwargs: Additional arguments passed to the parent RBLNModelConfig.
        """
        super().__init__(**kwargs)

        text_len = 512  # `WanPrompter(text_len=512)` pads every prompt to this many tokens.

        self.text_encoder = self.initialize_submodule_config(
            text_encoder,
            cls_name="RBLNQwen3ForCausalLMConfig",
            batch_size=1,
            max_seq_len=text_len,
            phases=["prefill"],
            output_hidden_states=True,
            force_kwargs=True,
        )
        self.transformer = self.initialize_submodule_config(
            transformer,
            cls_name="RBLNWanAudioModelConfig",
            batch_size=batch_size,
            text_len=text_len,
            max_inference_seconds=max_inference_seconds,
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNDACConfig",
            batch_size=batch_size,
            max_inference_seconds=max_inference_seconds,
        )

    @property
    def batch_size(self):
        return self.transformer.batch_size


__all__ = [
    "RBLNMossSoundEffectPipelineConfig",
]
