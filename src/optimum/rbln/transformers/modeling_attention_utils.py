from dataclasses import dataclass
from typing import TYPE_CHECKING

import rebel
import torch

from ..utils.logging import get_logger
from ..utils.runtime_utils import parse_byte_size, resolve_npu_or_none
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


def parse_memory_budget(value: int | float | str) -> int | float:
    if isinstance(value, str) and value.strip().endswith("%"):
        value = float(value.strip()[:-1]) / 100
    if isinstance(value, float):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"memory_budget fraction must be in (0, 1], got {value!r}.")
        return value
    return parse_byte_size(value)


def _resolve_memory_budget(memory_budget: int | float | str | None, available_total: int) -> int:
    if memory_budget is None:
        return available_total
    budget = parse_memory_budget(memory_budget)
    if isinstance(budget, float):
        return int(available_total * budget)
    if budget > available_total:
        raise ValueError(
            f"memory_budget ({budget} bytes) exceeds the target NPU's available DRAM ({available_total} bytes)."
        )
    return budget


class RBLNDecoderOnlyFlashAttentionMixin:
    @classmethod
    def estimate_num_kvcache_blocks(
        cls,
        compiled_models: list[rebel.RBLNCompiledModel],
        rbln_config: "RBLNDecoderOnlyModelConfig",
        device: torch.device,
    ) -> int:
        shards = cache_shards(compiled_models, rbln_config)
        stats = torch.rbln.mem_get_info_per_chiplet(device)
        room = {}
        for node, chiplet in cache_requests(shards, rbln_config, 1):
            node_total = stats[f"npu.{node}.total"]
            prefix = f"npu.{node}.chiplet.{chiplet}"
            budget = _resolve_memory_budget(rbln_config.memory_budget, node_total)
            room[node, chiplet] = budget * stats[f"{prefix}.total"] // node_total - stats[f"{prefix}.used"]
        return fit_cache_blocks(shards, rbln_config, room)
