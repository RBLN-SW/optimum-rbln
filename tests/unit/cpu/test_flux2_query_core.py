import pytest

from scripts.diagnostics import flux2_query_core as diagnostic


def test_query_chunks_preserve_every_token_and_original_head_order():
    import torch

    query = torch.arange(32, dtype=torch.float32).reshape(1, 1, 32, 1).expand(1, 512, 32, 128).contiguous()
    chunks = diagnostic.projected_chunks(query.flatten(2))
    assert len(chunks) == 4
    for index, chunk in enumerate(chunks):
        assert chunk.shape == (1, 512, 1024)
        assert torch.equal(chunk.view(1, 512, 8, 128), query[:, :, index * 8 : (index + 1) * 8])
    assert torch.equal(diagnostic.join_groups([chunk.view(1, 512, 8, 128) for chunk in chunks], "norm"), query)


@pytest.mark.parametrize("kind", ["norm", "rope"])
def test_missing_head_groups_cannot_count_as_full_query(kind):
    with pytest.raises(ValueError, match="four head groups"):
        diagnostic.join_groups([], kind)


def test_group_abi_rejects_wrong_head_extent():
    import torch

    groups = [torch.zeros(1, 512, 1, 128)] * 4
    with pytest.raises(ValueError, match="output ABI"):
        diagnostic.join_groups(groups, "norm")


def test_rotary_teacher_matches_pinned_native_function():
    import torch
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    normalized = torch.arange(1280, dtype=torch.float32).reshape(1, 5, 2, 128) / 1280
    angles = torch.arange(640, dtype=torch.float32).reshape(1, 5, 128) / 640
    cos, sin = angles.cos(), angles.sin()
    query = normalized.transpose(1, 2)
    expected, _ = apply_rotary_pos_emb(query, query, cos, sin)
    actual = diagnostic.rotary_reference(normalized, cos, sin)
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
