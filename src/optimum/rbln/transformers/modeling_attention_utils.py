import math
from collections import defaultdict
from typing import Optional, Tuple

import rebel

from ..utils.logging import get_logger
from ..utils.runtime_utils import get_available_dram, is_compiler_supports_buffer_resize
from .models.decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


logger = get_logger()


DEFAULT_FLASH_ATTN_PARTITION_LENGTH = 16_384
DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH = 32_768
MIN_FLASH_ATTN_MAX_SEQ_LEN = 64
MIN_FLASH_ATTN_PARTITION_LENGTH = 64
MAX_FLASH_ATTN_PARTITION_LENGTH = 32_768
MAX_SLIDING_WINDOW_SIZE = 32_768


def set_default_values(
    attn_impl: Optional[str] = None,
    kvcache_partition_len: Optional[int] = None,
    kvcache_block_size: Optional[int] = None,
    max_seq_len: Optional[int] = None,
) -> Tuple[str, int, int]:
    if attn_impl is None:
        attn_impl = "eager"

    if kvcache_partition_len is not None:
        if attn_impl == "eager":
            attn_impl = "flash_attn"
            logger.warning(
                "A non-null `kvcache_partition_len` was provided, but `attn_impl` was not explicitly set or "
                "set to 'eager'. Since KV cache partitioning is only supported with flash attention, "
                "`attn_impl` has been automatically switched to 'flash_attn'."
            )

    if kvcache_partition_len is None and attn_impl == "flash_attn":
        kvcache_partition_len = DEFAULT_FLASH_ATTN_PARTITION_LENGTH

    if kvcache_block_size is None:
        if attn_impl == "eager":
            kvcache_block_size = max_seq_len
        else:
            kvcache_block_size = kvcache_partition_len

    return attn_impl, kvcache_partition_len, kvcache_block_size


def validate_attention_method(attn_impl: str, kvcache_partition_len: int, kvcache_block_size: int, max_seq_len: int):
    if attn_impl not in ["eager", "flash_attn"]:
        raise ValueError(f"Unknown `attn_impl` : {attn_impl}. (Available : 'eager', 'flash_attn`)")

    ## Checking Constraints...
    # Constraint of eager attention:
    # - `max_seq_len` <= 32k

    # Constraints of flash attention:
    # 1. `max_seq_len` should be multiple of `partition_len`.
    # 2. 4k <= `partition_len` <= 32k.
    # 3. `max_seq_len` should be larger then 8k.
    if attn_impl == "eager" and max_seq_len > DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH:
        raise ValueError(
            f"`max_seq_len` is set to {max_seq_len}, "
            f"which exceeds the limit of {DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH} for 'eager' attention. "
            f"Please reduce the `max_seq_len` to {DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH} or lower,"
            " or consider switching `attn_impl` to 'flash_attn' for larger sequence lengths."
        )

    if attn_impl == "flash_attn":
        if max_seq_len // kvcache_partition_len < 2 or max_seq_len % kvcache_partition_len != 0:
            raise ValueError(
                f"`max_seq_len` ({max_seq_len}) must be a multiple of `kvcache_partition_len` ({kvcache_partition_len}) "
                f"when using 'flash_attn'. Please adjust either value to meet this requirement."
            )
        elif not (MIN_FLASH_ATTN_PARTITION_LENGTH <= kvcache_partition_len <= MAX_FLASH_ATTN_PARTITION_LENGTH):
            raise ValueError(
                f"`kvcache_partition_len` ({kvcache_partition_len}) is out of the supported range for 'flash_attn' "
                f"({MIN_FLASH_ATTN_PARTITION_LENGTH} <= `kvcache_partition_len` <= {MAX_FLASH_ATTN_PARTITION_LENGTH}). "
                f"Please provide a valid value within this range."
            )
        elif max_seq_len < MIN_FLASH_ATTN_MAX_SEQ_LEN:
            raise ValueError(
                f"`max_seq_len` ({max_seq_len}) is too small for 'flash_attn'. The minimum "
                f"supported value is {MIN_FLASH_ATTN_MAX_SEQ_LEN}. Please increase `max_seq_len` to meet "
                "this requirement, or consider switching `attn_impl` to 'eager' for shorter lengths."
            )

    if kvcache_block_size is not None:
        if attn_impl == "flash_attn" and kvcache_partition_len != kvcache_block_size:
            raise ValueError(
                f" When using 'flash attention', the `kvcache_block_size` ({kvcache_block_size})  "
                f"must always be set equal to the `kvcache_partition_len` {kvcache_partition_len}."
            )
        elif attn_impl == "eager" and kvcache_block_size != max_seq_len:
            raise ValueError(
                f" When using 'eager attention', the `kvcache_block_size` ({kvcache_block_size})  "
                f"must always be set equal to the `max_seq_len` {max_seq_len}."
            )


def validate_sliding_window(rbln_config: RBLNDecoderOnlyModelForCausalLMConfig):
    if rbln_config.sliding_window > MAX_SLIDING_WINDOW_SIZE - rbln_config.prefill_chunk_size:
        raise ValueError(
            f"Sliding window size ({rbln_config.sliding_window}) must be less than 32768 - prefill_chunk_size ({32768 - rbln_config.prefill_chunk_size})"
        )

    if rbln_config.cache_impl == "sliding_window" and rbln_config.use_attention_mask:
        raise ValueError("`use_attention_mask` must be set to False when `cache_impl` is set to 'sliding_window'.")


def align(x: int, nbytes: int) -> int:
    return int(math.ceil(x / nbytes) * nbytes)


def align_2MB(x: int) -> int:
    return align(x, 2**21)


def get_alloc_memory_by_key(compiled_models: dict[str, rebel.RBLNCompiledModel]) -> dict[str, int]:
    alloc_memory_by_key = defaultdict(int)
    # Get the actual memory allocation of each node by key
    for compiled_model in compiled_models.values():
        alloc_per_node_by_key = compiled_model.get_alloc_per_node_by_key()
        for key, memory_per_node in alloc_per_node_by_key.items():
            alloc_memory_by_key[key] += sum(memory_per_node)

    return alloc_memory_by_key


def format_byte_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024**2:
        return f"{nbytes / 1024:.2f} KB"
    elif nbytes < 1024**3:
        return f"{nbytes / 1024**2:.2f} MB"
    else:
        return f"{nbytes / 1024**3:.2f} GB"


class RBLNDecoderOnlyFlashAttentionMixin:
    @classmethod
    def set_kvcache_num_blocks_after_compilation(
        cls, compiled_models: dict[str, rebel.RBLNCompiledModel], rbln_config: RBLNDecoderOnlyModelForCausalLMConfig
    ):
        rbln_config.kvcache_num_blocks = cls.estimate_num_kvcache_blocks(
            compiled_models=compiled_models, rbln_config=rbln_config
        )
        if rbln_config.kvcache_num_blocks < rbln_config.num_min_blocks:
            raise ValueError(
                "Memory is not enought for full sequence length. "
                "Please consider decreasing `max_seq_len` to reduce the number of blocks."
            )
        cls.multiply_kv_cache_num_blocks(
            compiled_models=compiled_models, rbln_config=rbln_config, multiplier=rbln_config.kvcache_num_blocks
        )

    @classmethod
    def estimate_num_kvcache_blocks(
        cls,
        compiled_models: dict[str, rebel.RBLNCompiledModel],
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
        available_dram: Optional[int] = None,
    ) -> int:
        if available_dram is None:
            available_dram = get_available_dram(rbln_config.npu)

        if "prefill" not in rbln_config.phases:
            logger.warning(
                "Not estimating number of KV cache blocks since `prefill` phase is not in the `phases` list."
            )
            return 1

        num_node = rbln_config.tensor_parallel_size or 1
        alloc_per_node_without_dram = [0] * num_node

        for compiled_model in compiled_models.values():
            for key, alloc_per_node in compiled_model.get_alloc_per_node_by_key().items():
                if key == "DramTensor":
                    continue

                if len(alloc_per_node) != num_node:
                    alloc_per_node += [0] * (num_node - len(alloc_per_node))

                alloc_per_node_without_dram = [a + b for a, b in zip(alloc_per_node_without_dram, alloc_per_node)]

        remaining_dram_at_node: list[int] = [
            available_dram - without_dramtensor for without_dramtensor in alloc_per_node_without_dram
        ]

        # kvcache_tensor_sizes[key][node_id][chiplet_id] = alloc_size
        kvcache_tensor_sizes: dict[str, list[list[int]]] = compiled_models["prefill"].exp_get_dram_tensor_sizes()
        kvcache_meta_can_resize: dict[str, bool] = {
            kvcache_meta.name: kvcache_meta.can_resize for kvcache_meta in rbln_config.kvcache_metas
        }

        def get_updated_kvcache_tensor_sizes(
            kvcache_tensor_sizes: dict[str, list[list[int]]], multiplier: int
        ) -> dict[str, list[list[int]]]:
            # Get the updated KV cache tensor sizes by multiplying the multiplier
            # with considering attention type (full or sliding), and memory alignment.
            ret: dict[str, list[list[int]]] = {}
            for key, sizes_at_node in kvcache_tensor_sizes.items():
                m = multiplier if kvcache_meta_can_resize[key] else 1
                ret[key] = [
                    [align_2MB(size_at_chiplet * m) for size_at_chiplet in sizes_at_node_at_chiplet]
                    for sizes_at_node_at_chiplet in sizes_at_node
                ]
            return ret

        def check_memory_fits(multiplier: int) -> tuple[bool, list[int]]:
            # Check if the given multiplier fits in memory
            # Returns (fits: bool, kvcache_tensor_sizes_at_node: list[int])
            updated_kvcache_tensor_sizes = get_updated_kvcache_tensor_sizes(kvcache_tensor_sizes, multiplier)

            kvcache_tensor_sizes_at_node: list[int] = [0] * num_node
            for tensor_sizes_at_node in updated_kvcache_tensor_sizes.values():
                tensor_sizes_at_node: list[list[int]]
                for node_id, sizes_at_chiplet in enumerate(tensor_sizes_at_node):
                    sizes_at_chiplet: list[int]
                    kvcache_tensor_sizes_at_node[node_id] += sum(sizes_at_chiplet)

            fits = all(
                remaining_dram_at_node[node_id] >= kvcache_tensor_sizes_at_node[node_id] for node_id in range(num_node)
            )
            return fits, kvcache_tensor_sizes_at_node

        # Fast path: try maximum blocks first (most common case)
        fits, _ = check_memory_fits(rbln_config.num_full_blocks)
        if fits:
            # Best case: maximum blocks fit in memory
            return rbln_config.num_full_blocks

        # Slow path: binary search for optimal multiplier
        logger.debug(
            f"[KVCache] Not enough memory for {rbln_config.num_full_blocks} blocks. "
            f"Searching for optimal multiplier..."
        )

        left, right = 1, rbln_config.num_full_blocks - 1
        multiplier = 1  # Default to minimum if no valid multiplier found

        while left <= right:
            mid = (left + right) // 2
            fits, kvcache_tensor_sizes_at_node = check_memory_fits(mid)

            if fits:
                # Memory is sufficient, try larger multiplier
                multiplier = mid
                left = mid + 1
            else:
                # Memory is insufficient, try smaller multiplier
                logger.debug(
                    f"[KVCache] Not enough memory for {mid} blocks. Remaining DRAM: "
                    f"{[format_byte_size(remaining_dram) for remaining_dram in remaining_dram_at_node]}, "
                    f"KV cache tensor sizes: {[format_byte_size(size) for size in kvcache_tensor_sizes_at_node]}"
                )
                right = mid - 1

        return multiplier

    @classmethod
    def multiply_kv_cache_num_blocks(
        cls,
        compiled_models: dict[str, rebel.RBLNCompiledModel],
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
        multiplier: int,
    ):
        if not is_compiler_supports_buffer_resize():
            raise RuntimeError(
                "The installed version of rebel-compiler does not support automatic kv cache size determination. "
                "Please upgrade rebel-compiler to a version that supports this feature, "
                "or explicitly set 'kvcache_num_blocks' in rbln_config to manually specify the cache size."
            )

        for compiled_model in compiled_models.values():
            compiled_model.exp_multiply_buffer_size(
                {
                    kvcache_meta.name: multiplier
                    for kvcache_meta in rbln_config.kvcache_metas
                    if kvcache_meta.can_resize
                }
            )
