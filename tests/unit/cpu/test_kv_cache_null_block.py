import pytest
import torch

from optimum.rbln.transformers.cache_utils import CacheMeta, FullAttentionKVCacheMeta
from optimum.rbln.transformers.models.decoderonly.configuration_decoderonly import (
    RBLNDecoderOnlyModelForCausalLMConfig,
)
from optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils import RBLNPageTableManager


FLASH = {"max_seq_len": 8192, "attn_impl": "flash_attn", "kvcache_partition_len": 4096, "kvcache_block_size": 4096}
EAGER = {"max_seq_len": 4096, "attn_impl": "eager", "kvcache_block_size": 4096}


def _config(npu, **kwargs):
    return RBLNDecoderOnlyModelForCausalLMConfig(npu=npu, **kwargs)


def _serialized_meta(num_blocks, **extra):
    return {
        "name": "past_key_values_0",
        "layer_index": 0,
        "shape": [num_blocks, 2, 2, 1, 4096, 64],
        "layer_type": "full_attention",
        "is_auto": True,
        "dtype": "float16",
        **extra,
    }


@pytest.mark.parametrize(
    ("npu", "kwargs", "null_block_id", "num_min_blocks", "num_full_blocks"),
    [
        ("RBLN-CA25", {**FLASH, "batch_size": 1}, None, 2, 2),
        ("RBLN-CA25", {**FLASH, "batch_size": 2}, None, 3, 4),
        ("RBLN-CA25", {**EAGER, "batch_size": 1}, None, 1, 1),
        ("RBLN-CR13", {**FLASH, "batch_size": 1}, 0, 3, 3),
        ("RBLN-CR13", {**FLASH, "batch_size": 2}, 0, 3, 5),
        ("RBLN-CR13", {**FLASH, "batch_size": 4}, 0, 5, 9),
        ("RBLN-CR13", {**EAGER, "batch_size": 1}, 0, 2, 2),
        ("RBLN-CR13", {**EAGER, "batch_size": 2}, 0, 3, 3),
    ],
)
def test_block_counts_include_the_null_block(npu, kwargs, null_block_id, num_min_blocks, num_full_blocks):
    config = _config(npu, **kwargs)
    assert config.kvcache_null_block_id == null_block_id
    assert (config.num_min_blocks, config.num_full_blocks) == (num_min_blocks, num_full_blocks)


def test_meta_records_the_null_block_and_round_trips():
    config = _config("RBLN-CR13", **FLASH, batch_size=1)
    meta = FullAttentionKVCacheMeta.from_config("past_key_values_0", 0, 2, 64, "float16", config)
    assert meta.null_block_id == 0
    assert meta.shape[0] == 3
    assert CacheMeta.from_serialized(meta._prepare_for_serialization()).null_block_id == 0


def test_artifact_without_the_null_block_keeps_every_block():
    config = _config("RBLN-CR13", **FLASH, batch_size=1, kvcache_num_blocks=2, cache_metas=[_serialized_meta(2)])
    assert config.kvcache_null_block_id is None
    assert (config.num_min_blocks, config.num_full_blocks) == (2, 2)

    manager = RBLNPageTableManager(config)
    assert list(manager.free_block_pool) == [0, 1]
    block_tables, _ = manager.get_block_tables(torch.arange(0, 4608).unsqueeze(0), batch_idx=0)
    assert block_tables.tolist() == [0, 1]


def test_null_block_is_never_handed_out():
    meta = _serialized_meta(5, null_block_id=0)
    config = _config("RBLN-CR13", **FLASH, batch_size=2, kvcache_num_blocks=5, cache_metas=[meta])
    assert config.kvcache_null_block_id == 0

    manager = RBLNPageTableManager(config)
    assert list(manager.free_block_pool) == [1, 2, 3, 4]

    block_tables, _ = manager.get_block_tables(torch.arange(0, 512).unsqueeze(0), batch_idx=0)
    assert block_tables.tolist() == [1, 0]

    # both sequences reach the last partition: every data block is in use, none is the null block
    for batch_idx in range(2):
        manager.get_block_tables(torch.arange(0, 4608).unsqueeze(0), batch_idx=batch_idx)
    block_tables, _ = manager.get_block_tables(torch.tensor([[4608], [4608]]), batch_size=2, phase="decode")
    assert sorted(block_tables.flatten().tolist()) == [1, 2, 3, 4]
