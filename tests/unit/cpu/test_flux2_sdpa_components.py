import pytest

from scripts.diagnostics import flux2_sdpa_components as diagnostic


def test_expanded_kv_preserves_native_grouped_query_head_mapping():
    import torch

    query = torch.arange(32, dtype=torch.float32).reshape(1, 32, 1, 1).expand(1, 32, 512, 128)
    key = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1).expand(1, 8, 512, 128)
    for index, (q, k, v) in enumerate(diagnostic.expanded_chunks(query, key, key + 100)):
        assert torch.equal(q, query[:, index * 8 : (index + 1) * 8])
        assert k.shape == v.shape == q.shape
        assert torch.equal(k[0, :, 0, 0], torch.tensor([index * 2] * 4 + [index * 2 + 1] * 4))
        assert torch.equal(v, k + 100)


def test_component_handoff_preserves_blocked_softmax_key_and_query_axes():
    import torch

    logits = torch.arange(512 * 512, dtype=torch.float32).reshape(1, 1, 512, 512).expand(1, 8, 512, 512)
    probabilities = logits / 262144
    mask = torch.ones(1, 1, 512, 512, dtype=torch.bool)
    mask[..., 0, 65] = False
    query = torch.zeros(1, 8, 512, 128)
    key = query + 1
    value = query + 2
    context = query + 3

    def qk(q, k):
        assert q is query
        assert k is key
        return logits

    def softmax(blocked, blocked_mask):
        assert blocked[0, 0, 1, 0, 1] == logits[0, 0, 0, 65]
        assert blocked_mask[0, 0, 1, 0, 1] == 0
        return diagnostic.to_blocked(probabilities)

    def pv(p, v):
        assert torch.equal(p, probabilities)
        assert v is value
        return context

    actual = diagnostic.run_components({"qk": qk, "softmax": softmax, "pv": pv}, query, key, value, mask)
    assert all(torch.equal(a, b) for a, b in zip(actual, (logits, probabilities, context), strict=True))


def test_wrong_component_output_shape_is_rejected_before_the_next_runtime():
    import torch

    query = torch.zeros(1, 8, 512, 128)
    runtimes = {"qk": lambda q, k: torch.zeros(1, 8, 512, 128)}
    with pytest.raises(ValueError, match="qk output ABI"):
        diagnostic.run_components(runtimes, query, query, query, torch.ones(1, 1, 512, 512, dtype=torch.bool))
