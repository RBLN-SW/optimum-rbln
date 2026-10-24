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
