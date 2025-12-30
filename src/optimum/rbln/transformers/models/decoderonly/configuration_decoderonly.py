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

from typing import Any, Dict, List, Literal, Optional, Union, get_args

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger
from ...utils.rbln_quantization import RBLNQuantizationConfig
from .configuration_lora import RBLNLoRAConfig


logger = get_logger()

CacheImplType = Literal["static", "sliding_window", "hybrid"]
PhaseType = Literal["prefill", "image_prefill", "decode"]


class RBLNDecoderOnlyModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN decoder-only models.

    This class extends RBLNModelConfig with parameters specific to decoder-only transformer
    architectures optimized for RBLN devices. It controls aspects like attention implementation,
    KV cache management, and batching for inference.
    """

    _default_phases = ["prefill"]
    _default_logits_to_keep = 0

    def __init__(
        self,
        batch_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        use_inputs_embeds: Optional[bool] = None,
        use_attention_mask: Optional[bool] = None,
        use_position_ids: Optional[bool] = None,
        attn_impl: Optional[str] = None,
        kvcache_partition_len: Optional[int] = None,
        kvcache_block_size: Optional[int] = None,
        quantization: Optional[Union[Dict[str, Any], RBLNQuantizationConfig]] = None,
        lora_config: Optional[Union[Dict[str, Any], RBLNLoRAConfig]] = None,
        prefill_chunk_size: Optional[int] = None,
        kvcache_num_blocks: Optional[int] = None,
        decoder_batch_sizes: Optional[List[int]] = None,
        cache_impl: Optional[CacheImplType] = None,
        sliding_window: Optional[int] = None,
        sliding_window_layers: Optional[List[int]] = None,
        phases: Optional[List[PhaseType]] = None,
        logits_to_keep: Optional[int] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            max_seq_len (Optional[int]): The maximum sequence length supported by the model.
                If not provided, it attempts to infer from the model's configuration
                (`max_position_embeddings` or `n_positions`). Must be specified if not available
                in the model config.
            use_inputs_embeds (Optional[bool]): Whether to use input embeddings (`inputs_embeds`)
                directly instead of `input_ids`. Defaults to False. Requires the model to be
                compiled with this option enabled.
            use_attention_mask (Optional[bool]): Whether the model requires attention masks during
                inference. This is typically determined based on the target device and model
                architecture. Defaults are often set automatically based on the model and RBLN NPU.
            use_position_ids (Optional[bool]): Whether to use position IDs. Defaults to False.
            attn_impl (Optional[str]): Specifies the attention implementation to use.
                See the "Attention Implementation (`attn_impl`)" section below for details.
            kvcache_partition_len (Optional[int]): Defines the partition length for the KV cache
                when using "flash_attn". See the "KV Cache Partition Length (`kvcache_partition_len`)"
                section below for details.
            kvcache_block_size (Optional[int]): Sets the size (in number of tokens) of each block
                in the PagedAttention KV cache. See the "KV Cache Block Size (`kvcache_block_size`)"
                section below for details.
            quantization (Optional[Dict[str, Any]]): Configuration dictionary for applying model
                quantization. Specifies format, etc.
            lora_config (Optional[Union[Dict[str, Any], RBLNLoRAConfig]]): Configuration for LoRA
                (Low-Rank Adaptation) settings when using (multi-)LoRA support. Can be provided as
                a dictionary or an RBLNLoRAConfig instance. When provided, enables LoRA functionality
                for the model compilation. Defaults to None (no LoRA).
            prefill_chunk_size (Optional[int]): The chunk size used during the prefill phase for
                processing input sequences. Defaults to 128. Must be a positive integer
                divisible by 64. Affects prefill performance and memory usage.
            kvcache_num_blocks (Optional[int]): The total number of blocks to allocate for the
                PagedAttention KV cache. See the "KV Cache Number of Blocks (`kvcache_num_blocks`)"
                section below for details.
            decoder_batch_sizes (Optional[List[int]]): A list of batch sizes for which separate decoder models will be compiled.
                This allows the model to handle varying batch sizes efficiently during generation. If not specified,
                defaults to a list containing only the model's main batch size. When specifying multiple batch sizes:
                1) All values must be less than or equal to the main batch size.
                2) The list will be sorted in descending order (larger batch sizes first).
                3) If using multiple decoders, at least one batch size should match the main batch size.
            cache_impl (Optional[CacheImplType]): Specifies the KV cache implementation strategy. Defaults to "static".
                - "static": Uses a fixed-size global KV cache for all layers, suitable for standard attention patterns.
                - "sliding_window": Implements a sliding window KV cache, where each layer maintains a local cache of recent tokens.
                - "hybrid": Combines both static and sliding window approaches, allowing different layers to use different cache strategies.
                The choice affects memory usage and attention patterns. When using "sliding_window" or "hybrid",
                you must specify the `sliding_window` size and optionally `sliding_window_layers` for hybrid mode.
            sliding_window (Optional[int]): The size of the sliding window. Defaults to None.
            sliding_window_layers (Optional[List[int]]): The layers to use for the sliding window used in the hybrid model. Defaults to None.
            phases (Optional[List[PhaseType]]): The phases to compile the model for. Defaults to ["prefill"] if DecoderOnlyModel is used,
                ["prefill", "decode"] if DecoderOnlyModelForCausalLM is used.
            logits_to_keep (Optional[int]): The number of logits to keep for the decoder.  If set to 0, the decoder will keep all logits.
                Defaults to 0 if DecoderOnlyModel is used, 1 if DecoderOnlyModelForCausalLM is used.
            output_hidden_states (Optional[bool]): Whether to output the hidden states of the decoder. Defaults to False.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If `batch_size` is not a positive integer.
            ValueError: If `prefill_chunk_size` is not a positive integer divisible by 64.
            ValueError: If `max_seq_len` cannot be determined and is required.
            ValueError: If attention parameter constraints are violated (e.g., `max_seq_len` vs
                `kvcache_partition_len` for flash attention).


        Attention Implementation:
            `attn_impl` determines the underlying attention mechanism used by the model.

            - **`"eager"`** (Default if `kvcache_partition_len` is not set): Uses the standard PyTorch
                attention implementation. Suitable for sequences up to a certain limit (e.g., 32,768 tokens).
            - **`"flash_attn"`**: Utilizes an optimized Flash Attention implementation, beneficial for
                longer sequences and potentially faster execution. Requires `max_seq_len` to be at least
                8,192. If `kvcache_partition_len` is specified, `attn_impl` automatically defaults
                to `"flash_attn"`. When using `"flash_attn"`, `kvcache_block_size` must equal
                `kvcache_partition_len`.

            The choice impacts performance and memory usage, especially for long sequences.
            Constraints related to `max_seq_len` and `kvcache_partition_len` apply when using
            `"flash_attn"`.


        KV Cache Partition Length:
            `kvcache_partition_len` is relevant **only** when `attn_impl` is `"flash_attn"`.

            - It defines the length (number of tokens) of each partition within the Key-Value (KV) cache.
            - Must be between 4,096 and 32,768 (inclusive).
            - When using `"flash_attn"`, `max_seq_len` must be a multiple of `kvcache_partition_len`
                and at least twice its value (`max_seq_len >= 2 * kvcache_partition_len`).
            - If `attn_impl` is `"flash_attn"` and `kvcache_partition_len` is `None`, it defaults to
                16,384.


        KV Cache Number of Blocks:
            `kvcache_num_blocks` controls the total number of memory blocks allocated for the PagedAttention KV cache.
            Each block holds `kvcache_block_size` tokens of Key and Value states.

            - **Automatic Estimation (Default)**: If `kvcache_num_blocks` is `None`, the system estimates
                the maximum number of blocks that can fit into the available RBLN device memory. This
                calculation considers the model size (kernel memory), required buffer memory, the number
                of layers and heads, `kvcache_block_size`, tensor parallelism, and available RBLN NPU DRAM.
                This aims to maximize cache capacity for potentially better performance with long sequences
                or larger batches without manual tuning.
            - **Manual Setting**: You can explicitly set the number of blocks. This provides finer control
                but requires careful consideration of memory limits. Setting it too high may lead to
                compilation errors if it exceeds available memory. The system will issue warnings if your
                setting exceeds the estimated maximum.
            - **Performance Impact**: A larger number of blocks reduces the likelihood of cache eviction,
                which is beneficial for tasks involving many long sequences or large batch sizes, enabling
                higher throughput. However, allocating more blocks consumes more memory.
            - **Minimum Requirement**: The system requires a minimum number of blocks to function,
                calculated based on `max_seq_len`, `kvcache_block_size`, and `batch_size`. The number of
                allocated blocks must be sufficient to hold at least one full sequence length per item
                in the batch concurrently. The system will log warnings or raise errors if constraints
                are violated (e.g., if `kvcache_num_blocks` is less than `batch_size` when using Flash Attention).

            The optimal value depends on the specific model, task, hardware, and desired trade-off
            between performance and memory usage. The automatic estimation provides a robust starting point.
        """

        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.max_seq_len = max_seq_len
        self.use_inputs_embeds = use_inputs_embeds or False
        self.use_position_ids = use_position_ids or False
        self.use_attention_mask = use_attention_mask or False

        if self.use_position_ids and not self.use_attention_mask:
            raise ValueError("Position IDs should be used with attention mask.")

        self.quantization = quantization or {}
        if self.quantization and isinstance(self.quantization, dict):
            self.quantization = RBLNQuantizationConfig(**self.quantization)

        self.lora_config = lora_config
        if self.lora_config and isinstance(self.lora_config, dict):
            self.lora_config = RBLNLoRAConfig(**self.lora_config)

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

        self.attn_impl = attn_impl
        self.kvcache_partition_len = kvcache_partition_len
        self.kvcache_block_size = kvcache_block_size
        self.prefill_chunk_size = prefill_chunk_size or 128
        if self.prefill_chunk_size % 64 != 0 or self.prefill_chunk_size <= 0:
            raise ValueError("`prefill_chunk_size` must be a positive integer divisible by 64.")

        self.kvcache_num_blocks = kvcache_num_blocks
        self.cache_impl = cache_impl or "static"
        self.sliding_window = sliding_window
        self.sliding_window_layers = sliding_window_layers or []

        if phases is not None:
            self.validate_phases_type(phases)
        self.phases = phases or self._default_phases
        self.logits_to_keep = logits_to_keep or self._default_logits_to_keep
        if self.logits_to_keep is not None and self.logits_to_keep > 1:
            raise NotImplementedError("`logits_to_keep` > 1 is currently not supported for RBLN models.")

        self.output_hidden_states = output_hidden_states or False

        self.decoder_batch_sizes = None
        if "decode" in self.phases:
            self.decoder_batch_sizes = decoder_batch_sizes
            if self.decoder_batch_sizes is None:
                self.decoder_batch_sizes = [self.batch_size]

            if self.use_multiple_decoder:
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

    @staticmethod
    def validate_phases_type(phases: List[PhaseType]):
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
    def use_lora(self):
        return self.lora_config is not None

    @property
    def can_generate(self) -> bool:
        return "decode" in self.phases

    @property
    def nbits_per_param(self) -> int:
        if self.quantization:
            return self.quantization.nbits_per_param
        return 16


class RBLNDecoderOnlyModelForCausalLMConfig(RBLNDecoderOnlyModelConfig):
    """
    Configuration class for RBLN decoder-only models for Causal Language Modeling.

    This class extends RBLNModelConfig with parameters specific to decoder-only transformer
    architectures optimized for RBLN devices. It controls aspects like attention implementation,
    KV cache management, and batching for inference.
    """

    _default_phases = ["prefill", "decode"]
    _default_logits_to_keep = 1
