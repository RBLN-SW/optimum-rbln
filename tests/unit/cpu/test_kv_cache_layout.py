from types import SimpleNamespace

import pytest
import torch

from optimum.rbln.transformers.cache_utils import (
    CacheMeta,
    FullAttentionKVCacheMeta,
    SlidingWindowAttentionKVCacheMeta,
)
from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import (
    AttentionOp,
    SlidingWindowAttentionOp,
)
from optimum.rbln.transformers.models.qwen3_5.qwen3_5_architecture import Qwen3_5_CausalLMWrapper


def _cache_config(**overrides):
    config = {
        "kvcache_block_size": 1024,
        "is_auto_num_blocks": True,
        "num_full_blocks": 8,
        "kvcache_num_blocks": 0,
        "sliding_window": 512,
        "batch_size": 2,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def _attention_config(**overrides):
    config = {
        "use_attention_mask": False,
        "use_position_ids": False,
        "use_image_prefill": False,
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


def test_qwen3_5_splits_flat_states_by_layer_type():
    layer_types = ["full_attention", "linear_attention", "full_attention"]
    wrapper = SimpleNamespace(config=SimpleNamespace(layer_types=layer_types))
    states = [torch.full((1,), index) for index in range(4)]

    past_key_values, past_states = Qwen3_5_CausalLMWrapper._split_layer_states(wrapper, states)

    assert [None if cache is None else int(cache) for cache in past_key_values] == [0, None, 3]
    assert past_states[0] is None and past_states[2] is None
    assert [int(state) for state in past_states[1]] == [1, 2]


@pytest.mark.parametrize(
    "overrides, op_family, causal",
    [
        ({}, "flash_causal_attention_naive_", True),
        ({"use_attention_mask": True}, "flash_attention_naive_", False),
        ({"use_attention_mask": True, "use_position_ids": True}, "flash_causal_attention_naive_", True),
    ],
)
def test_attention_op_name(overrides, op_family, causal):
    op = AttentionOp(8, 64, 4, rbln_config=_attention_config(**overrides))
    for phase in ("prefill", "image_prefill", "decode"):
        op.phase = phase
        assert op.get_attn_op_name() == op_family + ("decode" if phase == "decode" else "prefill")
    assert op.is_causal is causal


@pytest.mark.parametrize(
    "overrides, phase, bidirectional",
    [
        ({}, "prefill", False),
        ({"use_attention_mask": True, "use_position_ids": True}, "prefill", True),
        ({"use_attention_mask": True, "use_position_ids": True, "use_image_prefill": True}, "prefill", False),
        ({"use_attention_mask": True, "use_position_ids": True, "use_image_prefill": True}, "image_prefill", True),
    ],
)
def test_bidirectional_prefill(overrides, phase, bidirectional):
    op = AttentionOp(8, 64, 4, rbln_config=_attention_config(**overrides))
    op.phase = phase
    assert op.is_bidirectional_prefill() is bidirectional


def test_sliding_window_attention_op_name():
    op = SlidingWindowAttentionOp(8, 64, 4, rbln_config=_attention_config())
    op.phase = "image_prefill"
    assert op.get_attn_op_name() == "sliding_window_attention_naive_prefill"
    op.phase = "decode"
    assert op.get_attn_op_name() == "sliding_window_attention_naive_decode"
