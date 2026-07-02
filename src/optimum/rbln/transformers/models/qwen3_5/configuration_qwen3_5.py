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

from typing import Any, List, Optional

from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig, RBLNDecoderOnlyModelForCausalLMConfig
from ..qwen3_vl.configuration_qwen3_vl import (
    RBLNQwen3VLForConditionalGenerationConfig,
    RBLNQwen3VLModelConfig,
    RBLNQwen3VLVisionModelConfig,
)


class RBLNQwen3_5ForCausalLMConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    """
    Configuration class for RBLN Qwen3.5 (text backbone) causal language models.

    Qwen3.5 is a hybrid decoder: most layers are `linear_attention` (GatedDeltaNet) and a
    minority are `full_attention` (gated softmax attention). Full-attention layers use the
    standard paged KV cache; linear-attention layers instead carry a `conv_state` and a
    `recurrent_state` (see `linear_attention_layers`). This config extends
    `RBLNDecoderOnlyModelForCausalLMConfig` only with `linear_attention_layers`; every other
    parameter behaves as in the base class.

    Example usage:
    ```python
    from optimum.rbln import RBLNQwen3_5ForCausalLM, RBLNQwen3_5ForCausalLMConfig

    config = RBLNQwen3_5ForCausalLMConfig(
        batch_size=1,
        max_seq_len=32768,
        tensor_parallel_size=4,
    )
    model = RBLNQwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-27B", export=True, rbln_config=config)
    ```
    """

    def __init__(self, linear_attention_layers: Optional[List[int]] = None, **kwargs: Any):
        """
        Args:
            linear_attention_layers (Optional[List[int]]): Indices of layers that use the
                GatedDeltaNet linear-attention token mixer (the remaining layers use full
                attention). Populated automatically from the HF `config.layer_types` during
                `_update_rbln_config`; rarely set by hand.
            kwargs: Additional arguments passed to `RBLNDecoderOnlyModelForCausalLMConfig`.
        """
        super().__init__(**kwargs)
        self.linear_attention_layers = linear_attention_layers or []


class RBLNQwen3_5TextModelConfig(RBLNDecoderOnlyModelConfig):
    """
    Configuration class for the bare RBLN Qwen3.5 text backbone (no LM head, text-only).

    See `RBLNQwen3_5ForCausalLMConfig` for the meaning of `linear_attention_layers`.
    """

    def __init__(self, linear_attention_layers: Optional[List[int]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.linear_attention_layers = linear_attention_layers or []


class RBLNQwen3_5VisionModelConfig(RBLNQwen3VLVisionModelConfig):
    """Vision encoder config for Qwen3.5 (same as Qwen3-VL vision: per-image `max_seq_len`)."""


class RBLNQwen3_5ModelConfig(RBLNQwen3VLModelConfig):
    """
    Configuration for the bare Qwen3.5 model (vision encoder + hybrid text, no LM head).

    Qwen3.5 is natively vision-language, so this is the multimodal model config. Adds
    `linear_attention_layers` (the GatedDeltaNet layer indices) on top of the Qwen3-VL
    multimodal config; the `visual` sub-config and `use_inputs_embeds=True` behave as in
    `RBLNQwen3VLModelConfig`.
    """

    def __init__(self, linear_attention_layers: Optional[List[int]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.linear_attention_layers = linear_attention_layers or []


class RBLNQwen3_5ForConditionalGenerationConfig(RBLNQwen3VLForConditionalGenerationConfig):
    """
    Configuration for `RBLNQwen3_5ForConditionalGeneration` (vision-language).

    Qwen3.5 pairs a Qwen3-VL-style vision encoder (no deepstack) with the hybrid Qwen3.5
    text backbone (`linear_attention` GatedDeltaNet layers + `full_attention` gated layers).
    The vision encoder output is injected into `inputs_embeds` (`use_inputs_embeds=True`,
    enforced by the parent). `linear_attention_layers` is populated automatically from
    `config.text_config.layer_types` during `_update_rbln_config`.

    Example usage:
    ```python
    from optimum.rbln import RBLNQwen3_5ForConditionalGeneration

    model = RBLNQwen3_5ForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3.5-...", export=True,
        rbln_config={"max_seq_len": 32768, "tensor_parallel_size": 4, "visual": {"max_seq_len": 6400}},
    )
    ```
    """
    
    # submodules = ["visual"]
    submodules = []

    def __init__(self, linear_attention_layers: Optional[List[int]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.linear_attention_layers = linear_attention_layers or []
