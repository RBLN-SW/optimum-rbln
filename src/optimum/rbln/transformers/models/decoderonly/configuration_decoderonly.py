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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal, get_args

from pydantic import Field, field_validator, model_validator

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger
from ...utils.rbln_quantization import RBLNQuantizationConfig
from .configuration_lora import RBLNLoRAConfig


logger = get_logger()

CacheImplType = Literal["static", "sliding_window", "hybrid"]
PhaseType = Literal["prefill", "image_prefill", "decode"]


@dataclass
class KVCacheMeta:
    """
    KVCacheMeta contains metadata describing the key-value (KV) cache tensor for a specific transformer layer.

    This is used during compilation and runtime on RBLN devices to manage memory and configure the
    static or dynamic characteristics of the cache implementation for decoder-only models.

    Attributes:
        name (str): Logical name of the KV cache tensor.
        layer_index (int): Index of the transformer layer corresponding to this cache.
        shape (list[int]): The 4D shape of the cache tensor:
            [num_blocks, num_heads, block_size, head_dim].
        layer_type (str): String describing the attention/cache algorithm.
        is_auto (bool): Whether the number of blocks is automatically determined.
        dtype (str): Data type of the cache buffer.
    """

    name: str
    layer_index: int
    shape: list[int]  # (num_blocks, num_heads, block_size(seq), head_dim)
    layer_type: str
    is_auto: bool
    dtype: str

    @property
    def compile_shape(self) -> list[int]:
        return [1, self.shape[1], self.shape[2], self.shape[3]] if self.can_resize else self.shape

    @property
    def can_resize(self) -> bool:
        return self.is_auto and self.layer_type == "full_attention"

    @property
    def num_blocks(self) -> int:
        return self.shape[0]

    @property
    def block_size(self) -> int:
        return self.shape[2]


class RBLNDecoderOnlyModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN decoder-only models.

    This class extends RBLNModelConfig with parameters specific to decoder-only transformer
    architectures optimized for RBLN devices. It controls aspects like attention implementation,
    KV cache management, and batching for inference.
    """

    supports_tp: ClassVar[bool] = True
    supports_quantization: ClassVar[bool] = True
    _default_phases: ClassVar[list[str]] = ["prefill"]
    _default_logits_to_keep: ClassVar[int] = 0

    batch_size: int = 1
    max_seq_len: int | None = None
    use_inputs_embeds: bool = False
    use_attention_mask: bool = False
    use_position_ids: bool = False
    attn_impl: str | None = None
    kvcache_partition_len: int | None = None
    kvcache_block_size: int | None = None
    quantization: RBLNQuantizationConfig | dict[str, Any] | None = None
    lora_config: RBLNLoRAConfig | dict[str, Any] | None = None
    prefill_chunk_size: int = 128
    kvcache_num_blocks: int = 0
    decoder_batch_sizes: list[int] | None = None
    cache_impl: CacheImplType = "static"
    sliding_window: int | None = None
    sliding_window_layers: list[int] = Field(default_factory=list)
    phases: list[PhaseType] | None = None
    logits_to_keep: int | None = None
    output_hidden_states: bool = False
    kvcache_metas: list[KVCacheMeta] = Field(default_factory=list)

    def __init__(self, **data: Any):
        # Set default phases and logits_to_keep from class variables if not provided
        if "phases" not in data or data["phases"] is None:
            data["phases"] = self.__class__._default_phases.copy()
        if "logits_to_keep" not in data or data["logits_to_keep"] is None:
            data["logits_to_keep"] = self.__class__._default_logits_to_keep

        # Handle quantization dict -> RBLNQuantizationConfig conversion
        quantization = data.get("quantization")
        if quantization and isinstance(quantization, dict):
            data["quantization"] = RBLNQuantizationConfig(**quantization)

        # Handle lora_config dict -> RBLNLoRAConfig conversion
        lora_config = data.get("lora_config")
        if lora_config and isinstance(lora_config, dict):
            data["lora_config"] = RBLNLoRAConfig(**lora_config)

        # Handle kvcache_metas list of dicts -> KVCacheMeta conversion
        kvcache_metas = data.get("kvcache_metas", [])
        if kvcache_metas and isinstance(kvcache_metas[0], dict):
            data["kvcache_metas"] = [KVCacheMeta(**meta) for meta in kvcache_metas]

        super().__init__(**data)

        # Validate LoRA adapters if LoRA is enabled
        if self.lora_config is not None:
            validation_results = self.lora_config.validate_adapter_weights()
            failed_adapters = [adapter_id for adapter_id, is_valid in validation_results.items() if not is_valid]

            if failed_adapters:
                raise ValueError(
                    f"Some LoRA adapters failed validation and may not be accessible at compile time: {failed_adapters}. "
                    "Please ensure all adapter weights are available and properly formatted."
                )

            logger.info(
                f"LoRA configuration initialized with {self.lora_config.num_adapters} adapters: "
                f"{self.lora_config.adapter_ids}. Max rank: {self.lora_config.max_lora_rank}"
            )

        # Setup decoder_batch_sizes if decode phase is enabled
        if "decode" in self.phases and self.decoder_batch_sizes is None:
            self.decoder_batch_sizes = [self.batch_size]

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        if v is None:
            return 1
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")
        return v

    @field_validator("prefill_chunk_size")
    @classmethod
    def validate_prefill_chunk_size(cls, v: int) -> int:
        if v % 64 != 0 or v <= 0:
            raise ValueError("`prefill_chunk_size` must be a positive integer divisible by 64.")
        return v

    @field_validator("logits_to_keep")
    @classmethod
    def validate_logits_to_keep(cls, v: int | None) -> int | None:
        if v is not None and v > 1:
            raise NotImplementedError("`logits_to_keep` > 1 is currently not supported for RBLN models.")
        return v

    @model_validator(mode="after")
    def validate_position_ids_with_attention_mask(self) -> "RBLNDecoderOnlyModelConfig":
        if self.use_position_ids and not self.use_attention_mask:
            raise ValueError("Position IDs should be used with attention mask.")
        return self

    @model_validator(mode="after")
    def validate_decoder_batch_sizes(self) -> "RBLNDecoderOnlyModelConfig":
        if self.decoder_batch_sizes is not None and self.use_multiple_decoder:
            if max(self.decoder_batch_sizes) > self.batch_size:
                raise ValueError(
                    f"Decoder batch size ({max(self.decoder_batch_sizes)}) must be less than or equal to the runtime batch size ({self.batch_size})."
                )
            if max(self.decoder_batch_sizes) < self.batch_size:
                logger.warning(
                    f"Maximum decoder batch size ({max(self.decoder_batch_sizes)}) is less than the model's batch size ({self.batch_size}). "
                    "Appending the model's batch size to the decoder batch size."
                )
                self.decoder_batch_sizes.append(self.batch_size)

            # Larger batch size should be at the beginning of the list.
            self.decoder_batch_sizes.sort(reverse=True)
        return self

    @staticmethod
    def validate_phases_type(phases: list[PhaseType]) -> None:
        if not isinstance(phases, list):
            raise ValueError("`phases` must be a list.")
        if not all(phase in get_args(PhaseType) for phase in phases):
            raise ValueError(f"All elements in `phases` must be of type `PhaseType`({get_args(PhaseType)}).")

    @property
    def use_global_attention(self) -> bool:
        return self.cache_impl in ["static", "hybrid"]

    @property
    def use_local_attention(self) -> bool:
        return self.cache_impl in ["sliding_window", "hybrid"]

    @property
    def use_multiple_decoder(self) -> bool:
        return isinstance(self.decoder_batch_sizes, list) and len(self.decoder_batch_sizes) > 1

    @property
    def use_lora(self) -> bool:
        return self.lora_config is not None

    @property
    def can_generate(self) -> bool:
        return "decode" in self.phases

    @property
    def use_image_prefill(self) -> bool:
        return "image_prefill" in self.phases

    @property
    def image_prefill_runtime_idx(self) -> int:
        return self.phases.index("image_prefill")

    @property
    def expected_compiled_model_names(self) -> list[str]:
        # ["prefill", "image_prefill", "decoder_batch_1", "decoder_batch_2", ...]
        if self.can_generate:
            return self.phases[: self.decoder_runtime_idx] + [
                f"decoder_batch_{batch_size}" for batch_size in self.decoder_batch_sizes
            ]
        else:
            return list(self.phases)

    @property
    def decoder_runtime_idx(self) -> int:
        if self.can_generate:
            return self.phases.index("decode")
        else:
            raise ValueError("`decode` phase is not in the phases.")

    @property
    def nbits_per_param(self) -> int:
        if self.quantization:
            return self.quantization.nbits_per_param
        return 16

    @property
    def is_auto_num_blocks(self) -> bool:
        """Returns True if kvcache_num_blocks will be automatically determined during compilation."""
        return self.kvcache_num_blocks == 0

    @property
    def num_full_blocks(self) -> int:
        return (self.max_seq_len // self.kvcache_block_size) * self.batch_size

    @property
    def num_min_blocks(self) -> int:
        if self.attn_impl == "flash_attn":
            return min(self.max_seq_len // self.kvcache_block_size + 1, self.num_full_blocks)
        return self.batch_size


class RBLNDecoderOnlyModelForCausalLMConfig(RBLNDecoderOnlyModelConfig):
    """
    Configuration class for RBLN decoder-only models for Causal Language Modeling.

    This class extends RBLNDecoderOnlyModelConfig with parameters specific to decoder-only transformer
    architectures optimized for RBLN devices. It controls aspects like attention implementation,
    KV cache management, and batching for inference.
    """

    _default_phases: ClassVar[list[str]] = ["prefill", "decode"]
    _default_logits_to_keep: ClassVar[int] = 1


def make_kvcache_meta(
    name: str,
    layer_index: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype: str,
    rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
) -> KVCacheMeta:
    """
    Factory function to create KVCacheMeta from config.

    This function is separate from KVCacheMeta class to avoid circular reference issues
    since it depends on RBLNDecoderOnlyModelForCausalLMConfig which is defined after KVCacheMeta.
    """
    assert len(rbln_config.compile_cfgs) == 0, "KVCacheMeta cannot be created from rbln_config with compile_cfgs"

    if rbln_config.sliding_window is not None and layer_index in rbln_config.sliding_window_layers:
        layer_type = "sliding_attention"
        block_size = rbln_config.sliding_window
        num_blocks = rbln_config.batch_size
        is_auto = False

    else:
        layer_type = "full_attention"
        block_size = rbln_config.kvcache_block_size

        if rbln_config.is_auto_num_blocks:
            num_blocks = rbln_config.num_full_blocks
            is_auto = True
        else:
            num_blocks = rbln_config.kvcache_num_blocks
            is_auto = False

    shape = [num_blocks, num_key_value_heads, block_size, head_dim]
    if num_blocks <= 0:
        raise ValueError("`num_blocks` must be greater than 0 when using KV cache.")

    return KVCacheMeta(
        name=name, layer_index=layer_index, shape=shape, layer_type=layer_type, is_auto=is_auto, dtype=dtype
    )
