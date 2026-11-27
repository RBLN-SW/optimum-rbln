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

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeModel,
    Qwen3_5MoeTextRotaryEmbedding,
    Qwen3_5MoeVisionModel,
    Qwen3_5MoeVisionPatchEmbed,
    Qwen3_5MoeVisionRotaryEmbedding,
)

from ...utils.moe import RBLNMoeLoadMixin
from ..qwen3_5.modeling_qwen3_5 import (
    RBLNQwen3_5ForCausalLM,
    RBLNQwen3_5ForConditionalGeneration,
    RBLNQwen3_5Model,
    RBLNQwen3_5TextModel,
    RBLNQwen3_5VisionModel,
)
from .qwen3_5_moe_architecture import Qwen3_5Moe_CausalLMWrapper, Qwen3_5Moe_LanguageModelWrapper


class RBLNQwen3_5MoeVisionModel(RBLNQwen3_5VisionModel):
    """Qwen3.5-MoE vision encoder for RBLN: the Qwen3.5 vision tower, built from the HF Qwen3_5Moe vision classes."""

    _vision_model_class = Qwen3_5MoeVisionModel
    _patch_embed_class = Qwen3_5MoeVisionPatchEmbed
    _vision_rotary_emb_class = Qwen3_5MoeVisionRotaryEmbedding


class RBLNQwen3_5MoeTextModel(RBLNMoeLoadMixin, RBLNQwen3_5TextModel):
    """The bare Qwen3.5-MoE text backbone (no LM head): the hybrid Qwen3.5 decoder with a sparse MoE MLP in
    every layer, run as the RBLN `custom_moe` op."""

    _decoder_wrapper_cls = Qwen3_5Moe_CausalLMWrapper


class RBLNQwen3_5MoeForCausalLM(RBLNMoeLoadMixin, RBLNQwen3_5ForCausalLM):
    """
    RBLNQwen3_5MoeForCausalLM is the text-only (causal language modeling) variant of Qwen3.5-MoE
    (Qwen3.5-35B-A3B, Qwen3.6-35B-A3B), optimized for RBLN NPUs. It runs the hybrid Qwen3.5 decoder —
    GatedDeltaNet `linear_attention` layers interleaved with gated `full_attention` layers — with the sparse
    MoE MLP of every layer on the RBLN `custom_moe` op, without the vision encoder.

    This model inherits from [`RBLNQwen3_5ForCausalLM`]. Check the superclass documentation for the generic
    methods the library implements for all its models.

    Examples:
        ```python
        from optimum.rbln import RBLNQwen3_5MoeForCausalLM

        model = RBLNQwen3_5MoeForCausalLM.from_pretrained(
            "Qwen/Qwen3.6-35B-A3B",
            export=True,
            rbln_config={
                "num_devices": 4,
                "kvcache_partition_len": 16384,
                "max_seq_len": 262144,
                "device": [0, 1, 2, 3],
            },
        )

        model.save_pretrained("compiled-qwen3.6-35b-a3b")
        ```
    """

    _decoder_wrapper_cls = Qwen3_5Moe_CausalLMWrapper


class RBLNQwen3_5MoeModel(RBLNMoeLoadMixin, RBLNQwen3_5Model):
    """The bare Qwen3.5-MoE model (vision encoder + hybrid MoE text backbone, no LM head)."""

    _decoder_wrapper_cls = Qwen3_5Moe_LanguageModelWrapper
    _config_class = Qwen3_5MoeConfig
    _rotary_emb_class = Qwen3_5MoeTextRotaryEmbedding
    _get_rope_index_func = Qwen3_5MoeModel.get_rope_index
    get_vision_position_ids = Qwen3_5MoeModel.get_vision_position_ids


class RBLNQwen3_5MoeForConditionalGeneration(RBLNMoeLoadMixin, RBLNQwen3_5ForConditionalGeneration):
    """
    RBLNQwen3_5MoeForConditionalGeneration is the vision-language Qwen3.5-MoE (Qwen3.5-35B-A3B,
    Qwen3.6-35B-A3B), optimized for RBLN NPUs. It pairs the Qwen3.5 vision encoder with the hybrid Qwen3.5
    text backbone whose every layer carries a sparse MoE MLP, run as the RBLN `custom_moe` op.

    This model inherits from [`RBLNQwen3_5ForConditionalGeneration`]. Check the superclass documentation for
    the generic methods the library implements for all its models.

    Important Note:
        This model includes a Large Language Model (LLM). For optimal performance, it is highly recommended to use
        tensor parallelism for the language model. This can be achieved by using the `rbln_config` parameter in the
        `from_pretrained` method. Refer to the `from_pretrained` documentation and the
        RBLNQwen3_5MoeForConditionalGenerationConfig class for details.

    Examples:
        ```python
        from optimum.rbln import RBLNQwen3_5MoeForConditionalGeneration

        model = RBLNQwen3_5MoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3.6-35B-A3B",
            export=True,
            rbln_config={
                "visual": {
                    "num_devices": 4,
                    "max_seq_len": 16384,
                    "device": [0, 1, 2, 3],
                },
                "num_devices": 4,
                "kvcache_partition_len": 16384,
                "max_seq_len": 262144,
                "device": [0, 1, 2, 3],
            },
        )

        model.save_pretrained("qwen3.6-35b-a3b")
        ```
    """

    _decoder_wrapper_cls = Qwen3_5Moe_LanguageModelWrapper
    _config_class = Qwen3_5MoeConfig
    _rotary_emb_class = Qwen3_5MoeTextRotaryEmbedding
    _get_rope_index_func = Qwen3_5MoeModel.get_rope_index
    get_vision_position_ids = Qwen3_5MoeModel.get_vision_position_ids


__all__ = [
    "RBLNQwen3_5MoeForCausalLM",
    "RBLNQwen3_5MoeForConditionalGeneration",
    "RBLNQwen3_5MoeModel",
    "RBLNQwen3_5MoeTextModel",
    "RBLNQwen3_5MoeVisionModel",
]
