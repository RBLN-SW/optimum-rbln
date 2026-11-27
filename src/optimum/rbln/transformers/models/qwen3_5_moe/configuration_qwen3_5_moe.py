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

from ..qwen3_5.configuration_qwen3_5 import (
    RBLNQwen3_5ForCausalLMConfig,
    RBLNQwen3_5ForConditionalGenerationConfig,
    RBLNQwen3_5ModelConfig,
    RBLNQwen3_5TextModelConfig,
    RBLNQwen3_5VisionModelConfig,
)


class RBLNQwen3_5MoeForCausalLMConfig(RBLNQwen3_5ForCausalLMConfig):
    """
    Configuration class for RBLN Qwen3.5-MoE (text backbone) causal language models.

    Qwen3.5-MoE (Qwen3.5-35B-A3B, Qwen3.6-35B-A3B) is the hybrid Qwen3.5 decoder with a sparse MoE MLP in
    every layer. The MoE layers take no option of their own, so this is `RBLNQwen3_5ForCausalLMConfig`
    (`gdn_chunk_size`, `linear_attention_layers`) under the MoE class name.

    Example usage:
    ```python
    from optimum.rbln import RBLNQwen3_5MoeForCausalLM, RBLNQwen3_5MoeForCausalLMConfig

    config = RBLNQwen3_5MoeForCausalLMConfig(
        batch_size=1,
        max_seq_len=32768,
        tensor_parallel_size=4,
    )
    model = RBLNQwen3_5MoeForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B", export=True, rbln_config=config)
    ```
    """


class RBLNQwen3_5MoeTextModelConfig(RBLNQwen3_5TextModelConfig):
    """Configuration class for the bare RBLN Qwen3.5-MoE text backbone (no LM head, text-only)."""


class RBLNQwen3_5MoeVisionModelConfig(RBLNQwen3_5VisionModelConfig):
    """Vision encoder config for Qwen3.5-MoE: the Qwen3.5 vision encoder, per-image `max_seq_len`."""


class RBLNQwen3_5MoeModelConfig(RBLNQwen3_5ModelConfig):
    """Configuration for the bare Qwen3.5-MoE model (vision encoder + hybrid MoE text, no LM head)."""


class RBLNQwen3_5MoeForConditionalGenerationConfig(RBLNQwen3_5ForConditionalGenerationConfig):
    """
    Configuration for `RBLNQwen3_5MoeForConditionalGeneration` (vision-language).

    Same options as `RBLNQwen3_5ForConditionalGenerationConfig`; the MoE layers take none of their own.

    Example usage:
    ```python
    from optimum.rbln import RBLNQwen3_5MoeForConditionalGeneration

    model = RBLNQwen3_5MoeForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3.6-35B-A3B", export=True,
        rbln_config={"max_seq_len": 32768, "tensor_parallel_size": 4, "visual": {"max_seq_len": 6400}},
    )
    ```
    """


__all__ = [
    "RBLNQwen3_5MoeForCausalLMConfig",
    "RBLNQwen3_5MoeForConditionalGenerationConfig",
    "RBLNQwen3_5MoeModelConfig",
    "RBLNQwen3_5MoeTextModelConfig",
    "RBLNQwen3_5MoeVisionModelConfig",
]
