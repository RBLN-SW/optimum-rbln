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
from ..qwen3.configuration_qwen3 import RBLNQwen3ForCausalLMConfig


class RBLNQwen3ASREncoderConfig(RBLNModelConfig):
    """
    Configuration class for RBLNQwen3ASREncoder.

    This configuration class stores the configuration parameters specific to the
    RBLN-optimized Qwen3-ASR audio encoder, which turns log-mel features into
    audio embeddings in the language model's embedding space.
    """

    def __init__(self, num_windows: int | None = None, **kwargs: Any):
        """
        Args:
            num_windows (int | None): Number of attention windows the graph encodes per pass,
                which is the batch axis of the compiled encoder. One window holds 104 tokens
                (8 seconds of audio) for the released model. This is a throughput knob, not a
                length limit: audio needing more windows than this is encoded in several passes.
                Defaults to the feature extractor's `chunk_length` (30 seconds) rounded up to a
                whole number of windows, which is 4.
            kwargs: Additional arguments passed to the parent `RBLNModelConfig`.
        """
        super().__init__(**kwargs)
        self.num_windows = num_windows


class RBLNQwen3ASRForConditionalGenerationConfig(RBLNQwen3ForCausalLMConfig):
    """
    Configuration class for RBLNQwen3ASRForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Qwen3-ASR models for speech recognition, which combine an audio
    encoder with a Qwen3 text decoder.
    """

    submodules = ["audio_tower"]

    def __init__(
        self,
        use_inputs_embeds: bool = True,
        audio_tower: RBLNModelConfig | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            use_inputs_embeds (bool): Whether or not to use `inputs_embeds` as input. Defaults to `True`.
            audio_tower (RBLNModelConfig | None): Configuration for the audio encoder component.
            kwargs: Additional arguments passed to the parent `RBLNQwen3ForCausalLMConfig`.

        Raises:
            ValueError: If `use_inputs_embeds` is False.
        """
        super().__init__(use_inputs_embeds=use_inputs_embeds, **kwargs)
        if not self.use_inputs_embeds:
            raise ValueError(
                "RBLNQwen3ASRForConditionalGenerationConfig does not allow `use_inputs_embeds` to be set to False, "
                "as audio embeddings are merged into the text embeddings before the decoder runs."
            )
        self.audio_tower = self.initialize_submodule_config(submodule_config=audio_tower)
