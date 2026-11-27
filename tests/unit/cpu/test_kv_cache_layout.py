from types import SimpleNamespace

import pytest

from optimum.rbln.transformers.cache_utils import (
    CacheMeta,
    FullAttentionKVCacheMeta,
    SlidingWindowAttentionKVCacheMeta,
)
from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import (
    AttentionOp,
    SlidingWindowAttentionOp,
)


def _cache_config(**overrides):
    config = {
        "kvcache_block_size": 1024,
        "is_auto_num_blocks": True,
        "num_full_blocks": 8,
        "kvcache_num_blocks": 0,
        "kvcache_null_block_id": None,
        "sliding_window": 512,
        "batch_size": 2,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def _attention_config(**overrides):
    config = {
        "use_attention_mask": False,
        "use_position_ids": False,
        "quantization": None,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def test_full_attention_meta_holds_key_and_value():
    meta = FullAttentionKVCacheMeta.from_config("past_key_values_0", 0, 4, 64, "float16", _cache_config())
    assert meta.shape == [8, 2, 4, 1, 1024, 64]
    assert meta.compile_shape == [1, 2, 4, 1, 1024, 64]
    assert (meta.num_blocks, meta.block_size) == (8, 1024)


def test_sliding_window_meta_holds_key_and_value():
    meta = SlidingWindowAttentionKVCacheMeta.from_config("past_key_values_1", 1, 4, 64, "float16", _cache_config())
    assert meta.shape == meta.compile_shape == [2, 2, 4, 1, 512, 64]
    assert (meta.num_blocks, meta.block_size, meta.can_resize) == (2, 512, False)


def test_serialized_separate_cache_meta_keeps_its_geometry():
    meta = CacheMeta.from_serialized(
        {
            "name": "past_key_values_1",
            "layer_index": 0,
            "shape": [8, 4, 1024, 64],
            "layer_type": "full_attention",
            "is_auto": False,
            "dtype": "float16",
        }
    )
    assert isinstance(meta, FullAttentionKVCacheMeta)
    assert (meta.num_blocks, meta.block_size, meta.compile_shape) == (8, 1024, [8, 4, 1024, 64])


QK_MASK = {"use_attention_mask": True}
KEY_MASK = {"use_attention_mask": True, "use_position_ids": True}


@pytest.mark.parametrize(
    ("overrides", "is_bidirectional", "causal"),
    [
        ({}, False, True),
        ({}, True, False),
        (QK_MASK, False, False),
        (QK_MASK, True, False),
        (KEY_MASK, False, True),
        (KEY_MASK, True, False),
    ],
)
def test_attention_op_direction(overrides, is_bidirectional, causal):
    op = AttentionOp(8, 64, 4, rbln_config=_attention_config(**overrides))
    for phase in ("prefill", "image_prefill", "decode"):
        op.phase = phase
        assert op.get_attn_op_name() == "gqa"
    assert op.is_causal(is_bidirectional) is causal


def test_sliding_window_attention_op_name():
    op = SlidingWindowAttentionOp(8, 64, 4, rbln_config=_attention_config())
    for phase in ("prefill", "image_prefill", "decode"):
        op.phase = phase
        assert op.get_attn_op_name() == "swa_v0"
