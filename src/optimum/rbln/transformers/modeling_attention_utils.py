import math
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Tuple

import rebel

from ..utils.logging import get_logger
from ..utils.runtime_utils import (
    get_available_dram,
    get_available_dram_per_chiplet,
    is_compiler_supports_buffer_resize,
    is_compiler_supports_chiplet_alloc,
)


if TYPE_CHECKING:
    from .models.decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


logger = get_logger()


DEFAULT_FLASH_ATTN_PARTITION_LENGTH = 16_384
DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH = 32_768
MIN_FLASH_ATTN_MAX_SEQ_LEN = 8192
MIN_FLASH_ATTN_PARTITION_LENGTH = 4096
MAX_FLASH_ATTN_PARTITION_LENGTH = 32_768
MAX_SLIDING_WINDOW_SIZE = 32_768


def _should_skip_attn_validation() -> bool:
    # Escape hatch read at call time so it can be toggled per-process (e.g. by external scripts
    # like rbln-executor or k-perf). When enabled, attention/sliding-window constraints are not
    # enforced — invalid configurations may still fail at compile or runtime.
    return os.environ.get("RBLN_SKIP_ATTN_VALIDATION", "0") == "1"


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
    if _should_skip_attn_validation():
        logger.warning(
            "Skipping `validate_attention_method` because `RBLN_SKIP_ATTN_VALIDATION=1`. "
            "Invalid configurations may still fail at compile or runtime."
        )
        return

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


def validate_sliding_window(rbln_config: "RBLNDecoderOnlyModelForCausalLMConfig"):
    if _should_skip_attn_validation():
        logger.warning(
            "Skipping `validate_sliding_window` because `RBLN_SKIP_ATTN_VALIDATION=1`. "
            "Invalid configurations may still fail at compile or runtime."
        )
        return

    if rbln_config.sliding_window > MAX_SLIDING_WINDOW_SIZE - rbln_config.prefill_chunk_size:
        raise ValueError(
            f"Sliding window size ({rbln_config.sliding_window}) must be less than {MAX_SLIDING_WINDOW_SIZE} - prefill_chunk_size ({MAX_SLIDING_WINDOW_SIZE - rbln_config.prefill_chunk_size})"
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
        cls, compiled_models: dict[str, rebel.RBLNCompiledModel], rbln_config: "RBLNDecoderOnlyModelForCausalLMConfig"
    ):
        rbln_config.kvcache_num_blocks = cls.estimate_num_kvcache_blocks(
            compiled_models=compiled_models, rbln_config=rbln_config
        )
        if rbln_config.kvcache_num_blocks < rbln_config.num_min_blocks:
            raise ValueError(
                "Memory is not enough for full sequence length. "
                "Please consider decreasing `max_seq_len` to reduce the number of blocks."
            )
        cls.multiply_kv_cache_num_blocks(
            compiled_models=compiled_models, rbln_config=rbln_config, multiplier=rbln_config.kvcache_num_blocks
        )

    @classmethod
    def estimate_num_kvcache_blocks(
        cls,
        compiled_models: dict[str, rebel.RBLNCompiledModel],
        rbln_config: "RBLNDecoderOnlyModelForCausalLMConfig",
        available_dram: Optional[int] = None,
    ) -> int:
        if "prefill" not in rbln_config.phases:
            logger.warning(
                "Not estimating number of KV cache blocks since `prefill` phase is not in the `phases` list."
            )
            return 1

        # Device DRAM is partitioned per chiplet, so a block count that fits the node
        # total can still OOM a single chiplet; the search below bounds blocks by the
        # tightest chiplet.
        alloc_without_dram, kvcache_tensor_sizes, available_per_chiplet, chiplets = (
            cls._collect_chiplet_kvcache_inputs(compiled_models, rbln_config, available_dram)
        )
        return cls._search_num_kvcache_blocks(
            rbln_config, alloc_without_dram, kvcache_tensor_sizes, available_per_chiplet, chiplets
        )

    @classmethod
    def _collect_chiplet_kvcache_inputs(
        cls,
        compiled_models: dict[str, rebel.RBLNCompiledModel],
        rbln_config: "RBLNDecoderOnlyModelForCausalLMConfig",
        available_dram: Optional[int] = None,
    ) -> Tuple[dict[Tuple[int, int], int], dict[str, list[list[int]]], int, set[Tuple[int, int]]]:
        # Returns non-KV alloc, KV sizes, per-bucket DRAM budget, and the (node, chiplet)
        # buckets to check. ATOM reports one chiplet, so it shares the per-chiplet path.
        alloc_without_dram: dict[Tuple[int, int], int] = defaultdict(int)
        chiplets: set[Tuple[int, int]] = set()

        if is_compiler_supports_chiplet_alloc():
            for compiled_model in compiled_models.values():
                for key, alloc_per_chiplet in compiled_model.get_alloc_per_chiplet_by_key().items():
                    if key == "DramTensor":
                        continue
                    for node_id, sizes_at_chiplet in enumerate(alloc_per_chiplet):
                        for chiplet_id, size in enumerate(sizes_at_chiplet):
                            alloc_without_dram[(node_id, chiplet_id)] += size
                            chiplets.add((node_id, chiplet_id))

            # kvcache_tensor_sizes[key][node_id][chiplet_id] = alloc_size
            kvcache_tensor_sizes: dict[str, list[list[int]]] = compiled_models["prefill"].exp_get_dram_tensor_sizes()
            for sizes_at_node in kvcache_tensor_sizes.values():
                for node_id, sizes_at_chiplet in enumerate(sizes_at_node):
                    for chiplet_id in range(len(sizes_at_chiplet)):
                        chiplets.add((node_id, chiplet_id))

            num_chiplets = max((chiplet_id for _, chiplet_id in chiplets), default=0) + 1
            available_per_chiplet = get_available_dram_per_chiplet(num_chiplets, rbln_config.npu)
            return alloc_without_dram, kvcache_tensor_sizes, available_per_chiplet, chiplets

        # Legacy compiler exposes only node totals, so collapse each node into one bucket
        # with the whole-node budget; the search then reduces to the node-level check.
        for compiled_model in compiled_models.values():
            for key, alloc_per_node in compiled_model.get_alloc_per_node_by_key().items():
                if key == "DramTensor":
                    continue
                for node_id, size in enumerate(alloc_per_node):
                    alloc_without_dram[(node_id, 0)] += size
                    chiplets.add((node_id, 0))

        # Sum the per-chiplet KV sizes into the single bucket to match alloc's shape.
        raw_kvcache: dict[str, list[list[int]]] = compiled_models["prefill"].exp_get_dram_tensor_sizes()
        kvcache_tensor_sizes = {}
        for key, sizes_at_node in raw_kvcache.items():
            kvcache_tensor_sizes[key] = [[sum(sizes_at_chiplet)] for sizes_at_chiplet in sizes_at_node]
            for node_id in range(len(sizes_at_node)):
                chiplets.add((node_id, 0))

        available_per_chiplet = available_dram if available_dram is not None else get_available_dram(rbln_config.npu)
        return alloc_without_dram, kvcache_tensor_sizes, available_per_chiplet, chiplets

    @classmethod
    def _search_num_kvcache_blocks(
        cls,
        rbln_config: "RBLNDecoderOnlyModelForCausalLMConfig",
        alloc_without_dram: dict[Tuple[int, int], int],
        kvcache_tensor_sizes: dict[str, list[list[int]]],
        available_per_chiplet: int,
        chiplets: set[Tuple[int, int]],
    ) -> int:
        remaining_dram_at_chiplet: dict[Tuple[int, int], int] = {
            key: available_per_chiplet - alloc_without_dram.get(key, 0) for key in chiplets
        }
        kvcache_meta_can_resize: dict[str, bool] = {
            kvcache_meta.name: kvcache_meta.can_resize for kvcache_meta in rbln_config.kvcache_metas
        }

        def kvcache_sizes_at_chiplet(multiplier: int) -> dict[Tuple[int, int], int]:
            # Resize multiplier applies only to resizable tensors; 2MB-aligned.
            sizes: dict[Tuple[int, int], int] = defaultdict(int)
            for key, sizes_at_node in kvcache_tensor_sizes.items():
                m = multiplier if kvcache_meta_can_resize[key] else 1
                for node_id, sizes_at_chiplet in enumerate(sizes_at_node):
                    for chiplet_id, size in enumerate(sizes_at_chiplet):
                        sizes[(node_id, chiplet_id)] += align_2MB(size * m)
            return sizes

        def check_memory_fits(multiplier: int) -> Tuple[bool, dict[Tuple[int, int], int]]:
            # Fits only if every chiplet bucket has room.
            kvcache_sizes = kvcache_sizes_at_chiplet(multiplier)
            fits = all(remaining_dram_at_chiplet[key] >= kvcache_sizes.get(key, 0) for key in chiplets)
            return fits, kvcache_sizes

        # Fast path: try maximum blocks first (most common case)
        fits, _ = check_memory_fits(rbln_config.num_full_blocks)
        if fits:
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
            fits, kvcache_sizes = check_memory_fits(mid)

            if fits:
                multiplier = mid
                left = mid + 1
            else:
                tightest = min(
                    (remaining_dram_at_chiplet[key] - kvcache_sizes.get(key, 0) for key in chiplets),
                    default=0,
                )
                logger.debug(
                    f"[KVCache] Not enough memory for {mid} blocks. "
                    f"Tightest chiplet headroom: {format_byte_size(tightest)}"
                )
                right = mid - 1

        return multiplier

    @classmethod
    def multiply_kv_cache_num_blocks(
        cls,
        compiled_models: dict[str, rebel.RBLNCompiledModel],
        rbln_config: "RBLNDecoderOnlyModelForCausalLMConfig",
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
