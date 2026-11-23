from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from optimum.rbln.transformers import modeling_attention_utils
from optimum.rbln.transformers.device_kv_cache import (
    DecoderCacheRuntime,
    DeviceKVCache,
    allocator_reserved,
    cache_shape_override,
    fit_cache_blocks,
)


MIB = 1 << 20


def meta(name, dynamic=True):
    return SimpleNamespace(name=name, can_resize=dynamic, shape=[32 if dynamic else 3, 2], dtype="float16")


def config(*metas):
    return SimpleNamespace(cache_metas=metas, num_min_blocks=4, num_full_blocks=32)


def test_shared_symbol_only_on_resizable_cache_inputs():
    cfg = config(meta("kv0"), meta("conv", False), meta("kv1"))
    inputs = SimpleNamespace(
        input_info=[
            ("tokens", [1, 8], "int64"),
            ("kv0", [1, 2], "float16"),
            ("conv", [3, 2], "float16"),
            ("kv1", [1, 2], "float16"),
        ]
    )
    symbols, dims = cache_shape_override(cfg, inputs)
    (symbol,) = symbols.values()
    assert (symbol.input_idx, symbol.dim_idx, symbol.max_size) == (1, 0, 32)
    assert dims[1] == dims[3] == [("symbol", symbol.sym_name), ("const", 2)]
    assert dims[2] == [("const", 3), ("const", 2)]


def test_prefill_and_decode_share_cache_with_inputs_after_cache():
    cache = DeviceKVCache(config(meta("kv"), meta("conv", False)), torch.device("cpu"), 5)
    runtime = Mock(_index_to_input_name={0: "tokens", 1: "kv", 2: "conv", 3: "mask"})
    prefill, decode = DecoderCacheRuntime(runtime, cache), DecoderCacheRuntime(runtime, cache)
    mask = torch.ones(1)
    prefill(mask, mask)
    decode(mask, mask=mask)
    assert runtime.run.call_count == 2
    for call in runtime.run.call_args_list:
        assert call.kwargs.keys() == {"tokens", "mask", "kv", "conv"}
        assert call.kwargs["kv"] is cache.tensors["kv"]
        assert call.kwargs["conv"] is cache.tensors["conv"]
        assert call.kwargs["mask"] is mask


def test_cache_pool_segments_can_shrink_as_block_count_grows():
    assert allocator_reserved([MIB, MIB]) == 2 * MIB
    assert allocator_reserved([10 * MIB]) == 20 * MIB
    assert allocator_reserved([10 * MIB + 1]) == 12 * MIB
    cfg = config(meta("kv"), meta("state", False))
    shards = {"kv": [(0, 0, MIB // 2), (1, 0, MIB)], "state": [(0, 0, 4096)]}
    room = {(0, 0): 14 * MIB, (1, 0): 28 * MIB}
    assert fit_cache_blocks(shards, cfg, room) == 24
    with pytest.raises(ValueError, match="Insufficient memory"):
        fit_cache_blocks(shards, cfg, {**room, (1, 0): MIB})


@pytest.mark.parametrize(
    ("budget", "expected"), [(None, 16), (0.875, 12), ("87.5%", 12), ("56MB", 12), (56 * MIB, 12)]
)
def test_auto_cache_capacity_uses_free_memory_on_each_chiplet(monkeypatch, budget, expected):
    shards = {"kv": [(0, 0, MIB), (0, 1, MIB)]}
    monkeypatch.setattr(modeling_attention_utils, "cache_shards", Mock(return_value=shards))
    stats = {
        "npu.0.total": 64 * MIB,
        "npu.0.chiplet.0.total": 32 * MIB,
        "npu.0.chiplet.0.used": 0,
        "npu.0.chiplet.0.free": 32 * MIB,
        "npu.0.chiplet.1.total": 32 * MIB,
        "npu.0.chiplet.1.used": 16 * MIB,
        "npu.0.chiplet.1.free": 16 * MIB,
    }
    monkeypatch.setattr(torch.rbln, "mem_get_info_per_chiplet", Mock(return_value=stats))
    cfg = config(meta("kv"))
    cfg.memory_budget = budget
    blocks = modeling_attention_utils.RBLNDecoderOnlyFlashAttentionMixin.estimate_num_kvcache_blocks(
        [], cfg, torch.device("rbln:0")
    )
    assert blocks == expected


def test_fixed_cache_uses_full_block_count():
    cfg = config(meta("kv", False))
    assert fit_cache_blocks({"kv": [(0, 0, MIB)]}, cfg, {(0, 0): 4 * MIB}) == cfg.num_full_blocks
