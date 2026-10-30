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
    """
    Configuration class for RBLNLlama4ForCausalLM.

    This configuration class is a subclass of RBLNDecoderOnlyModelForCausalLMConfig and accepts
    the same arguments. Refer to RBLNDecoderOnlyModelForCausalLMConfig for the full list of
    options (for example `batch_size`, `max_seq_len`, `attn_impl`, `kvcache_partition_len`,
    `num_devices`, and `use_attention_mask`).
    """


class RBLNLlama4VisionModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLNLlama4VisionModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Llama 4 vision encoders.
    """

    def __init__(self, batch_size: int | None = None, **kwargs: Any):
        """
        Args:
            batch_size (int | None): The number of image tiles encoded in one call. Larger inputs are
                processed in several calls and smaller inputs are padded. Defaults to 4 when used in
                RBLNLlama4ForConditionalGeneration and 1 otherwise.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If `batch_size` is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = _validate_batch_size(batch_size)


class RBLNLlama4ForConditionalGenerationConfig(RBLNModelConfig):
    """
    Configuration class for RBLNLlama4ForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Llama 4 models for multimodal conditional generation tasks
    that combine vision and language processing capabilities.
    """

    submodules = ["vision_model", "language_model"]

    def __init__(
        self,
        batch_size: int | None = None,
        vision_model: RBLNModelConfig | dict[str, Any] | None = None,
        language_model: RBLNModelConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size for inference. Defaults to 1.
            vision_model (RBLNModelConfig | dict[str, Any] | None): Configuration for the vision encoder component.
                Its `batch_size` sets the maximum number of image tiles processed per call.
                Its `num_devices` sets vision tensor parallelism independently of the language model.
                If not provided, default settings will be used.
            language_model (RBLNModelConfig | dict[str, Any] | None): Configuration for the language model component.
                This can include settings specific to the language model, such as `max_seq_len` or `num_devices`.
                The language model always uses `inputs_embeds` as input.
                If not provided, default settings will be used.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If `batch_size` is not a positive integer, or if `language_model` sets
                `use_inputs_embeds=False`.
        """
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
