import pytest

from scripts.diagnostics import flux2_block0_selective as diagnostic


def test_both_original_inputs_are_required_for_attribution():
    cases = [
        {"name": "short_component", "matches_original": True, "repeat_exact": True},
        {"name": "native_valid512", "matches_original": True, "repeat_exact": True},
    ]
    assert diagnostic.faithful_cases(cases)
    cases[1]["matches_original"] = False
    assert not diagnostic.faithful_cases(cases)


def test_one_matching_input_is_insufficient():
    cases = [{"name": "short_component", "matches_original": True, "repeat_exact": True}]
    with pytest.raises(ValueError, match="baseline cases"):
        diagnostic.faithful_cases(cases)


def test_matching_endpoints_with_unstable_repeats_reject_attribution():
    cases = [
        {"name": "short_component", "matches_original": True, "repeat_exact": False},
        {"name": "native_valid512", "matches_original": True, "repeat_exact": True},
    ]
    assert not diagnostic.faithful_cases(cases)


@pytest.mark.parametrize("invalid", ["broadcastable_head", "wrong_dtype", "wrong_endpoint"])
def test_producer_outputs_reject_silent_broadcasts_and_abi_drift(monkeypatch, invalid):
    import torch

    monkeypatch.setattr(diagnostic, "VARIANTS", {"query": 10})
    endpoint = torch.zeros(1, 512, 2560)
    query = torch.zeros(1, 512, 32, 128)
    reference = {"original_full_NPU_h1": endpoint, "CPU_features": [None] * 10 + [query]}
    actual_endpoint, actual_query = endpoint, query
    if invalid == "broadcastable_head":
        actual_query = query[:, :, :1]
    elif invalid == "wrong_dtype":
        actual_query = query.half()
    else:
        actual_endpoint = endpoint[:, :1]
    with pytest.raises(ValueError, match="output ABI"):
        diagnostic.validate_outputs("query", (actual_endpoint, actual_query), reference)


def test_producer_outputs_accept_native_query_dimensions(monkeypatch):
    import torch

    monkeypatch.setattr(diagnostic, "VARIANTS", {"query": 10})
    endpoint = torch.zeros(1, 512, 2560)
    query = torch.zeros(1, 512, 32, 128)
    reference = {"original_full_NPU_h1": endpoint, "CPU_features": [None] * 10 + [query]}
    diagnostic.validate_outputs("query", (endpoint, query), reference)
