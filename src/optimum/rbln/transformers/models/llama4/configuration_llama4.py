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

from typing import Any

from ....configuration_utils import RBLNModelConfig
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


def _validate_batch_size(batch_size: int | None) -> int | None:
    if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
    return batch_size


class RBLNLlama4ForCausalLMConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    """Configuration class for RBLN Llama 4 causal language models.

    This class is an alias of RBLNDecoderOnlyModelForCausalLMConfig.
    """


class RBLNLlama4VisionModelConfig(RBLNModelConfig):
    """Compilation configuration for the Llama 4 vision encoder."""

    def __init__(self, batch_size: int | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.batch_size = _validate_batch_size(batch_size)


class RBLNLlama4ForConditionalGenerationConfig(RBLNModelConfig):
    """Configuration for Llama 4 image and text generation.

    Decoder compilation options such as max_seq_len and attn_impl belong in
    language_model. Image tiles are encoded one at a time on one device.
    """

    submodules = ["vision_model", "language_model"]

    def __init__(
        self,
        batch_size: int | None = None,
        vision_model: RBLNModelConfig | dict[str, Any] | None = None,
        language_model: RBLNModelConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.batch_size = _validate_batch_size(batch_size) or 1

        self.vision_model = self.initialize_submodule_config(submodule_config=vision_model)
        if language_model is None or isinstance(language_model, dict):
            # batch_size is only a default for the language model; an explicit value in the dict wins.
            language_model = {"batch_size": self.batch_size, **(language_model or {})}
        self.language_model = self.initialize_submodule_config(
            submodule_config=language_model,
            use_inputs_embeds=True,  # Llama4 language model must use inputs_embeds as input
            force_kwargs=True,
        )

        # force_kwargs only reconciles dict inputs; check instances passed directly.
        if isinstance(self.language_model, RBLNModelConfig) and not self.language_model.use_inputs_embeds:
            raise ValueError("Llama 4 conditional generation requires language_model.use_inputs_embeds=True.")

    @property
    def use_inputs_embeds(self):
        return self.language_model.use_inputs_embeds


__all__ = [
    "RBLNLlama4ForCausalLMConfig",
    "RBLNLlama4VisionModelConfig",
    "RBLNLlama4ForConditionalGenerationConfig",
]
