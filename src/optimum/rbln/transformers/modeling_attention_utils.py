from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import rebel
import torch

from ..utils.logging import get_logger
from ..utils.runtime_utils import get_available_dram_per_chiplet, parse_byte_size, resolve_npu_or_none
from .device_kv_cache import (
    cache_requests,
    cache_shards,
    fit_cache_blocks,
)


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

    @classmethod
    def estimate_num_kvcache_blocks(
        cls,
        compiled_models: dict[str, rebel.RBLNCompiledModel],
        rbln_config: "RBLNDecoderOnlyModelConfig",
    ) -> int:
        if "prefill" not in rbln_config.phases:
            logger.warning(
                "Not estimating number of KV cache blocks since `prefill` phase is not in the `phases` list."
            )
            return 1

        shards = cache_shards(compiled_models, rbln_config)
        used: dict[tuple[int, int], int] = defaultdict(int)
        for model in compiled_models.values():
            for allocation in model.get_alloc_per_chiplet_by_key().values():
                for node, chiplets in enumerate(allocation):
                    for chiplet, size in enumerate(chiplets):
                        used[node, chiplet] += size

        units = used.keys() | cache_requests(shards, rbln_config, 1).keys()
        num_chiplets = max(chiplet for _, chiplet in units) + 1
        available = get_available_dram_per_chiplet(num_chiplets, rbln_config.npu) * num_chiplets
        budget = _resolve_memory_budget(rbln_config.memory_budget, available) // num_chiplets
        return fit_cache_blocks(shards, rbln_config, {unit: budget - used[unit] for unit in units})

    @classmethod
    def estimate_live_cache_blocks(cls, compiled_models, config, device):
        stats = torch.rbln.mem_get_info_per_chiplet(device)
        totals = {}
        for key, value in stats.items():
            parts = key.split(".")
            if len(parts) == 5 and parts[2] == "chiplet" and parts[-1] == "total":
                totals[int(parts[1]), int(parts[3])] = value
        if not totals:
            raise ValueError("The driver did not report per-chiplet DRAM for runtime_memory_budget.")
        room = {}
        for (node, chiplet), total in totals.items():
            node_total = sum(size for (n, _), size in totals.items() if n == node)
            budget = _resolve_memory_budget(config.runtime_memory_budget, node_total)
            used = stats[f"npu.{node}.chiplet.{chiplet}.used"]
            room[node, chiplet] = budget * total // node_total - used
        return fit_cache_blocks(cache_shards(compiled_models, config), config, room)
