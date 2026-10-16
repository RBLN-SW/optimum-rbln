from types import SimpleNamespace

import pytest
import torch

from optimum.rbln.transformers.cache_utils import CacheMeta, FullAttentionKVCacheMeta
from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import group_kv_caches_by_layer
from optimum.rbln.transformers.models.decoderonly.modeling_decoderonly import RBLNDecoderOnlyModel


def _cache_config(**overrides):
    config = {
        "kvcache_block_size": 1024,
        "is_auto_num_blocks": True,
        "num_full_blocks": 8,
        "kvcache_num_blocks": 0,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def _attention_config(**overrides):
    config = {
        "attn_impl": "flash_attn",
        "cache_impl": "static",
        "use_attention_mask": False,
        "use_image_prefill": False,
        "quantization": None,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


@pytest.mark.parametrize(
    "holds_key_and_value, shape, compile_shape",
    [
        (True, [8, 2, 4, 1, 1024, 64], [1, 2, 4, 1, 1024, 64]),
        (False, [8, 4, 1024, 64], [1, 4, 1024, 64]),
    ],
)
def test_full_attention_meta_shape(holds_key_and_value, shape, compile_shape):
    meta = FullAttentionKVCacheMeta.from_config(
        "past_key_values_0", 0, 4, 64, "float16", _cache_config(), holds_key_and_value=holds_key_and_value
    )
    assert meta.shape == shape
    assert meta.compile_shape == compile_shape
    assert (meta.num_blocks, meta.block_size) == (8, 1024)


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


@pytest.mark.parametrize("holds_key_and_value", [True, False])
def test_group_kv_caches_by_layer(holds_key_and_value):
    num_layers = 3
    caches_per_layer = 1 if holds_key_and_value else 2
    metas = [
        FullAttentionKVCacheMeta.from_config(
            f"past_key_values_{index}",
            index // caches_per_layer,
            4,
            64,
            "float16",
            _cache_config(),
            holds_key_and_value=holds_key_and_value,
        )
        for index in range(num_layers * caches_per_layer)
    ]
    caches = [torch.full((1,), index) for index in range(len(metas))]

    grouped = group_kv_caches_by_layer(caches, SimpleNamespace(cache_metas=metas), num_layers)

    assert [len(layer) for layer in grouped] == [caches_per_layer] * num_layers
    assert [int(cache) for layer in grouped for cache in layer] == list(range(len(metas)))
    with pytest.raises(ValueError):
        group_kv_caches_by_layer(caches[:-1], SimpleNamespace(cache_metas=metas), num_layers)


@pytest.mark.parametrize(
    "overrides, packed",
    [
        ({}, True),
        ({"attn_impl": "eager"}, False),
        ({"cache_impl": "hybrid"}, False),
        ({"use_attention_mask": True}, False),
        ({"use_image_prefill": True}, False),
        ({"quantization": SimpleNamespace(kv_caches="fp8")}, False),
        ({"quantization": SimpleNamespace(kv_caches="fp16")}, True),
    ],
)
def test_use_packed_kv_cache(overrides, packed):
    assert RBLNDecoderOnlyModel.use_packed_kv_cache(_attention_config(**overrides)) is packed
