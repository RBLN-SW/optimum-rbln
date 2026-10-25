import pytest

from scripts.diagnostics import flux2_sdpa_core as diagnostic


def test_query_groups_keep_the_matching_kv_head_groups():
    import torch

    query = torch.arange(32, dtype=torch.float32).reshape(1, 32, 1, 1).expand(1, 32, 512, 128)
    key = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1).expand(1, 8, 512, 128)
    value = key + 100
    groups = diagnostic.attention_chunks(query, key, value)
    assert len(groups) == 4
    for index, (q, k, v) in enumerate(groups):
        assert torch.equal(q, query[:, index * 8 : (index + 1) * 8])
        assert torch.equal(k, key[:, index * 2 : (index + 1) * 2])
        assert torch.equal(v, value[:, index * 2 : (index + 1) * 2])


def test_missing_kv_heads_rejects_the_partial_attention_contract():
    import torch

    query = torch.zeros(1, 32, 512, 128)
    key = torch.zeros(1, 2, 512, 128)
    with pytest.raises(ValueError, match="native Q/K/V"):
        diagnostic.attention_chunks(query, key, key)
