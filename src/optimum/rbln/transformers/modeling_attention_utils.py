import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import rebel

from ..utils.logging import get_logger
from ..utils.runtime_utils import get_available_dram_per_chiplet, parse_byte_size, resolve_npu_or_none


if TYPE_CHECKING:
    from .models.decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig


logger = get_logger()


@dataclass(frozen=True)
class AttentionLimits:
    """Bounds on the extent of the KV cache's dynamic axis for one NPU family.

    The bounded parameter differs per attention mode, because that is what each mode makes the
    cache's dynamic axis: `max_seq_len` for eager, `kvcache_partition_len` for flash attention,
    and `sliding_window` for a sliding-window cache.
    """

    name: str
    max_eager_seq_len: int
    min_flash_partition_len: int
    max_flash_partition_len: int
    default_flash_partition_len: int
    max_sliding_window: int

    @property
    def min_flash_max_seq_len(self) -> int:
        # Flash attention needs at least two partitions.
        return 2 * self.min_flash_partition_len


ATOM_ATTENTION_LIMITS = AttentionLimits(
    name="ATOM",
    max_eager_seq_len=32_768,
    min_flash_partition_len=1_024,
    max_flash_partition_len=32_768,
    default_flash_partition_len=16_384,
    max_sliding_window=32_768,
)

REBEL_ATTENTION_LIMITS = AttentionLimits(
    name="REBEL",
    max_eager_seq_len=16_384,
    min_flash_partition_len=1_024,
    max_flash_partition_len=16_384,
    default_flash_partition_len=8_192,
    max_sliding_window=32_767,
)


def get_attention_limits(npu: str | None = None) -> AttentionLimits:
    """Attention limits of the target NPU. ATOM accepts twice the extent REBEL does.

    With no NPU to name — compiling on a host without one and without `npu` pinned on the config —
    the wider ATOM limits apply and the compiler stays the backstop. A named-but-unknown NPU is a
    different case and raises: inheriting the wider limits there is exactly the silent compiler
    abort this guard exists to prevent.
    """
    npu = resolve_npu_or_none(npu)
    if npu is None:
        return ATOM_ATTENTION_LIMITS
    if npu.startswith("RBLN-CR"):
        return REBEL_ATTENTION_LIMITS
    if npu.startswith("RBLN-CA"):
        return ATOM_ATTENTION_LIMITS
    raise ValueError(f"Unknown npu name: {npu}")


def set_default_values(
    attn_impl: str | None = None,
    kvcache_partition_len: int | None = None,
    kvcache_block_size: int | None = None,
    max_seq_len: int | None = None,
    prefill_chunk_size: int | None = None,
    npu: str | None = None,
) -> tuple[str, int | None, int | None, int]:
    if attn_impl is None:
        attn_impl = "eager"

    npu = resolve_npu_or_none(npu)

    if prefill_chunk_size is None:
        # RBLN-CR NPUs use a larger prefill chunk for better prefill performance.
        prefill_chunk_size = 512 if "RBLN-CR" in (npu or "") else 128
    if prefill_chunk_size % 64 != 0 or prefill_chunk_size <= 0:
        raise ValueError("`prefill_chunk_size` must be a positive integer divisible by 64.")

    if kvcache_partition_len is not None:
        if attn_impl == "eager":
            attn_impl = "flash_attn"
            logger.warning(
                "A non-null `kvcache_partition_len` was provided, but `attn_impl` was not explicitly set or "
                "set to 'eager'. Since KV cache partitioning is only supported with flash attention, "
                "`attn_impl` has been automatically switched to 'flash_attn'."
            )

    if kvcache_partition_len is None and attn_impl == "flash_attn":
        kvcache_partition_len = get_attention_limits(npu).default_flash_partition_len

    if kvcache_block_size is None:
        if attn_impl == "eager":
            kvcache_block_size = max_seq_len
        else:
            kvcache_block_size = kvcache_partition_len

    return attn_impl, kvcache_partition_len, kvcache_block_size, prefill_chunk_size


def validate_attention_method(
    attn_impl: str,
    kvcache_partition_len: int,
    kvcache_block_size: int,
    max_seq_len: int,
    npu: str | None = None,
) -> None:
    if attn_impl not in ["eager", "flash_attn"]:
        raise ValueError(f"Unknown `attn_impl` : {attn_impl}. (Available : 'eager', 'flash_attn`)")

    limits = get_attention_limits(npu)

    if attn_impl == "eager" and max_seq_len > limits.max_eager_seq_len:
        raise ValueError(
            f"`max_seq_len` is set to {max_seq_len}, "
            f"which exceeds the {limits.name} limit of {limits.max_eager_seq_len} for 'eager' attention. "
            f"Please reduce the `max_seq_len` to {limits.max_eager_seq_len} or lower,"
            " or consider switching `attn_impl` to 'flash_attn' for larger sequence lengths."
        )

    if attn_impl == "flash_attn":
        if max_seq_len // kvcache_partition_len < 2 or max_seq_len % kvcache_partition_len != 0:
            raise ValueError(
                f"`max_seq_len` ({max_seq_len}) must be a multiple of `kvcache_partition_len` ({kvcache_partition_len}) "
                f"when using 'flash_attn'. Please adjust either value to meet this requirement."
            )
        elif not (limits.min_flash_partition_len <= kvcache_partition_len <= limits.max_flash_partition_len):
            raise ValueError(
                f"`kvcache_partition_len` ({kvcache_partition_len}) is out of the {limits.name} supported range "
                f"for 'flash_attn' ({limits.min_flash_partition_len} <= `kvcache_partition_len` <= "
                f"{limits.max_flash_partition_len}). Please provide a valid value within this range."
            )
        elif max_seq_len < limits.min_flash_max_seq_len:
            raise ValueError(
                f"`max_seq_len` ({max_seq_len}) is too small for 'flash_attn'. The minimum "
                f"supported value is {limits.min_flash_max_seq_len}. Please increase `max_seq_len` to meet "
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


def validate_sliding_window(rbln_config: "RBLNDecoderOnlyModelConfig") -> None:
    if rbln_config.sliding_window is None or rbln_config.prefill_chunk_size is None:
        raise ValueError("`sliding_window` and `prefill_chunk_size` must be set to validate the sliding window.")
    limits = get_attention_limits(rbln_config.npu)
    max_sliding_window = limits.max_sliding_window - rbln_config.prefill_chunk_size
    if rbln_config.sliding_window > max_sliding_window:
        raise ValueError(
            f"Sliding window size ({rbln_config.sliding_window}) must be at most {max_sliding_window} on "
            f"{limits.name} (`max_sliding_window` {limits.max_sliding_window} - `prefill_chunk_size` "
            f"{rbln_config.prefill_chunk_size})."
        )

    if rbln_config.cache_impl == "sliding_window" and rbln_config.use_attention_mask:
        raise ValueError("`use_attention_mask` must be set to False when `cache_impl` is set to 'sliding_window'.")


def align(x: int, nbytes: int) -> int:
    return int(math.ceil(x / nbytes) * nbytes)


def align_2MB(x: int) -> int:
    return align(x, 2**21)


def get_alloc_memory_by_key(compiled_models: dict[str, rebel.RBLNCompiledModel]) -> dict[str, int]:
    alloc_memory_by_key: defaultdict[str, int] = defaultdict(int)
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


def _resolve_memory_budget(memory_budget: object | None, available_total: int) -> int:
    """Resolve `memory_budget` to usable DRAM bytes (system reserve excluded), capped at available_total.

    None -> available_total; a float in (0, 1] (or a "80%" string) -> that fraction of it;
    int/"10GB"/"512MB" -> parsed bytes. `available_total` is the device-wide available DRAM after
    the per-chiplet system reserve.
    """
    if memory_budget is None:
        return available_total
    fraction = None
    if isinstance(memory_budget, float):
        fraction = memory_budget
    elif isinstance(memory_budget, str) and memory_budget.strip().endswith("%"):
        fraction = float(memory_budget.strip()[:-1]) / 100
    if fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"memory_budget fraction must be in (0, 1] (or (0%, 100%]), got {memory_budget!r}.")
        budget = int(available_total * fraction)
    elif isinstance(memory_budget, (int, str)):
        budget = parse_byte_size(memory_budget)
    else:
        raise ValueError(
            f"memory_budget must be None, a float, an int or a string, got {type(memory_budget).__name__}."
        )
    if budget > available_total:
        raise ValueError(
            f"memory_budget ({budget} bytes) exceeds the target NPU's available DRAM ({available_total} bytes)."
        )
    return budget


class RBLNDecoderOnlyFlashAttentionMixin:
    @classmethod
    def set_kvcache_num_blocks_after_compilation(
        cls, compiled_models: dict[str, rebel.RBLNCompiledModel], rbln_config: "RBLNDecoderOnlyModelConfig"
    ):
        def _log_memory_usage(compiled_models: dict[str, rebel.RBLNCompiledModel], prefix: str):
            if not logger.isEnabledFor(logging.DEBUG):
                return
            for phase, compiled_model in compiled_models.items():
                logger.debug(f"{prefix} Memory usage of compiled_model[{phase}]:")
                for key, alloc_per_chiplet in compiled_model.get_alloc_per_chiplet_by_key().items():
                    logger.debug(
                        f"  {key}: {[[format_byte_size(size) for size in sizes_at_chiplet] for sizes_at_chiplet in alloc_per_chiplet]}"
                    )

                logger.debug(f"{prefix} DramTensor sizes in compiled_model[{phase}]:")
                logger.debug("Please note that the sizes are not aligned. (alignment is not considered)")
                for key, sizes_at_node in compiled_model.exp_get_dram_tensor_sizes().items():
                    logger.debug(f"  {key}:")
                    for node_id, sizes_at_chiplet in enumerate(sizes_at_node):
                        logger.debug(f"    - node {node_id}: {[format_byte_size(size) for size in sizes_at_chiplet]}")

        _log_memory_usage(compiled_models, "Before adjusting kvcache_num_blocks:")

        rbln_config.kvcache_num_blocks = cls.estimate_num_kvcache_blocks(
            compiled_models=compiled_models, rbln_config=rbln_config
        )
        if rbln_config.kvcache_num_blocks < rbln_config.num_min_blocks:
            raise ValueError(
                f"Insufficient memory for the required KV cache: only {rbln_config.kvcache_num_blocks} blocks "
                f"can be allocated, but at least {rbln_config.num_min_blocks} blocks are required. Consider "
                f"reducing `max_seq_len` or `batch_size` (currently {rbln_config.batch_size}) to lower the "
                f"memory requirement."
            )
        cls.multiply_kv_cache_num_blocks(
            compiled_models=compiled_models, rbln_config=rbln_config, multiplier=rbln_config.kvcache_num_blocks
        )

        _log_memory_usage(compiled_models, "After adjusting kvcache_num_blocks:")

    @classmethod
    def estimate_num_kvcache_blocks(
        cls,
        compiled_models: dict[str, rebel.RBLNCompiledModel],
        rbln_config: "RBLNDecoderOnlyModelConfig",
        current_blocks: int = 1,
    ) -> int:
        # `current_blocks` is the block count the loaded buffers already hold: 1 at compile time,
        # or `rbln_config.kvcache_num_blocks` when re-estimating for an already-resized artifact.
        if "prefill" not in rbln_config.phases:
            logger.warning(
                "Not estimating number of KV cache blocks since `prefill` phase is not in the `phases` list."
            )
            return 1

        # Device DRAM is partitioned per chiplet, so a block count that fits the node
        # total can still OOM a single chiplet; the search below bounds blocks by the
        # tightest chiplet.
        alloc_without_dram, kvcache_tensor_sizes, available_per_chiplet, chiplets = (
            cls._collect_chiplet_kvcache_inputs(compiled_models, rbln_config)
        )
        return cls._search_num_kvcache_blocks(
            rbln_config, alloc_without_dram, kvcache_tensor_sizes, available_per_chiplet, chiplets, current_blocks
        )

    @classmethod
    def _collect_chiplet_kvcache_inputs(
        cls,
        compiled_models: dict[str, rebel.RBLNCompiledModel],
        rbln_config: "RBLNDecoderOnlyModelConfig",
    ) -> tuple[dict[tuple[int, int], int], dict[str, list[list[int]]], int, set[tuple[int, int]]]:
        # Returns non-KV alloc, KV sizes, per-chiplet DRAM budget, and the (node, chiplet)
        # buckets to check. ATOM reports one chiplet, so it shares the per-chiplet path.
        alloc_without_dram: dict[tuple[int, int], int] = defaultdict(int)
        chiplets: set[tuple[int, int]] = set()

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
        available_total = get_available_dram_per_chiplet(num_chiplets, rbln_config.npu) * num_chiplets
        budget = _resolve_memory_budget(rbln_config.memory_budget, available_total)
        available_per_chiplet = budget // num_chiplets
        return alloc_without_dram, kvcache_tensor_sizes, available_per_chiplet, chiplets

    @classmethod
    def _search_num_kvcache_blocks(
        cls,
        rbln_config: "RBLNDecoderOnlyModelConfig",
        alloc_without_dram: dict[tuple[int, int], int],
        kvcache_tensor_sizes: dict[str, list[list[int]]],
        available_per_chiplet: int,
        chiplets: set[tuple[int, int]],
        current_blocks: int = 1,
    ) -> int:
        remaining_dram_at_chiplet: dict[tuple[int, int], int] = {
            key: available_per_chiplet - alloc_without_dram.get(key, 0) for key in chiplets
        }

        def check_memory_fits(multiplier: int) -> tuple[bool, dict[tuple[int, int], int]]:
            # Fits only if every chiplet bucket has room.
            kvcache_sizes = cls._kvcache_bytes_per_chiplet(
                kvcache_tensor_sizes, rbln_config, multiplier, current_blocks
            )
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
    def _kvcache_bytes_per_chiplet(
        cls,
        kvcache_tensor_sizes: dict[str, list[list[int]]],
        rbln_config: "RBLNDecoderOnlyModelConfig",
        num_blocks: int,
        current_blocks: int = 1,
    ) -> dict[tuple[int, int], int]:
        # Per-(node, chiplet) kv-cache bytes at `num_blocks`. `kvcache_tensor_sizes` reflects the
        # buffers' current block count (`current_blocks`), so a resizable tensor's per-block bytes
        # are size // current_blocks, rescaled to num_blocks; others are fixed. Each 2MB-aligned
        # after scaling, so bytes grow non-linearly.
        can_resize = {meta.name: meta.can_resize for meta in rbln_config.cache_metas}
        sizes: dict[tuple[int, int], int] = defaultdict(int)
        for key, sizes_at_node in kvcache_tensor_sizes.items():
            resizable = can_resize[key]
            for node_id, sizes_at_chiplet in enumerate(sizes_at_node):
                for chiplet_id, size in enumerate(sizes_at_chiplet):
                    scaled = size // current_blocks * num_blocks if resizable else size
                    sizes[(node_id, chiplet_id)] += align_2MB(scaled)
        return sizes

    @classmethod
    def _required_memory_at(
        cls,
        compiled_models: dict[str, rebel.RBLNCompiledModel],
        rbln_config: "RBLNDecoderOnlyModelConfig",
        num_blocks: int,
    ) -> int:
        """Total device-wide kv-cache DRAM (bytes) at `num_blocks`, with 2MB alignment applied.

        Works for any current block size: the buffers' block count is `rbln_config.kvcache_num_blocks`
        (0 for the unresized compile baseline is treated as 1). Bytes are not linear in `num_blocks`
        because alignment is applied after scaling.
        """
        kvcache_tensor_sizes = compiled_models["prefill"].exp_get_dram_tensor_sizes()
        current_blocks = rbln_config.kvcache_num_blocks or 1
        return sum(
            cls._kvcache_bytes_per_chiplet(kvcache_tensor_sizes, rbln_config, num_blocks, current_blocks).values()
        )

    @classmethod
    def multiply_kv_cache_num_blocks(
        cls,
        compiled_models: dict[str, rebel.RBLNCompiledModel],
        rbln_config: "RBLNDecoderOnlyModelConfig",
        multiplier: int,
    ):
        for compiled_model in compiled_models.values():
            compiled_model.exp_multiply_buffer_size(
                {cache_meta.name: multiplier for cache_meta in rbln_config.cache_metas if cache_meta.can_resize}
            )

    @classmethod
    def rescale_kvcache_num_blocks(
        cls,
        compiled_models: dict[str, rebel.RBLNCompiledModel],
        rbln_config: "RBLNDecoderOnlyModelConfig",
        target: int,
    ):
        """Resize an already-compiled artifact's kv-cache to `target` blocks.

        The saved buffers hold `per_block * current` bytes, where `current` is the block
        count the artifact currently carries (`rbln_config.kvcache_num_blocks`): the count
        resolved at compile time, or the target of an earlier rescale. Scaling by the
        rational ratio `target / current` yields `per_block * target` exactly, so
        `exp_rescale_buffer_size` never rejects the ratio for a block-size-1 baseline.
        """
        current = rbln_config.kvcache_num_blocks
        if current <= 0:
            raise ValueError(
                f"Cannot rescale kv-cache: the artifact's current kvcache_num_blocks ({current}) "
                "is not a resolved positive block count."
            )
        if target < rbln_config.num_min_blocks:
            raise ValueError(
                f"kvcache_num_blocks={target} is below the minimum required for the full sequence "
                f"length (num_min_blocks={rbln_config.num_min_blocks})."
            )
        for compiled_model in compiled_models.values():
            if not hasattr(compiled_model, "exp_rescale_buffer_size"):
                raise RuntimeError(
                    "The installed rebel-compiler does not support post-compilation kv-cache "
                    "resizing (`exp_rescale_buffer_size`). Please upgrade rebel-compiler. "
                    "See https://docs.rbln.ai/about_atom/release_note.html"
                )
        if target > current:
            logger.warning(
                f"Requested kvcache_num_blocks={target} exceeds the artifact's current block count "
                f"({current}), which was sized to fit device DRAM; the model may fail to allocate at "
                "runtime. Proceeding without a device-fit check."
            )
        if target > rbln_config.num_full_blocks:
            logger.warning(
                f"Requested kvcache_num_blocks={target} exceeds num_full_blocks "
                f"({rbln_config.num_full_blocks}), the blocks needed to cover the full batch at "
                "max_seq_len; the excess blocks are never used."
            )
        for compiled_model in compiled_models.values():
            compiled_model.exp_rescale_buffer_size(
                {cache_meta.name: (target, current) for cache_meta in rbln_config.cache_metas if cache_meta.can_resize}
            )
