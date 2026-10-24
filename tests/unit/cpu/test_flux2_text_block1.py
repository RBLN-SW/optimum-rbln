import json

import pytest
import torch

from scripts.diagnostics import flux2_text_block1 as diagnostic


def test_all_native_outputs_and_auxiliary_result_required_for_attribution():
    baseline = {i: torch.tensor([float(i)]) for i in range(37)}
    values = tuple(baseline.values()) + tuple(torch.tensor([0.0]) for _ in range(6)) + (baseline[2].clone(),)
    identity = diagnostic.identity_checks(values, baseline)
    assert all(identity["native37"])
    assert identity["auxiliary_final_matches_native_block"]
    changed = list(values)
    changed[27] = changed[27] + 1
    assert not all(diagnostic.identity_checks(changed, baseline)["native37"])
    changed = list(values)
    changed[43] = changed[43] + 1
    assert not diagnostic.identity_checks(changed, baseline)["auxiliary_final_matches_native_block"]


def test_auxiliary_outputs_do_not_replace_missing_native_layers():
    with pytest.raises(ValueError, match="output ABI"):
        diagnostic.identity_checks([torch.zeros(1)] * 43, {i: torch.zeros(1) for i in range(37)})


def test_unscheduled_execution_rejected_before_reading_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("BUILDKITE_BUILD_ID", raising=False)
    assert diagnostic.main(["--execute", "--artifacts", str(tmp_path / "missing"), "--output", str(tmp_path)]) == 1
    result = json.loads((tmp_path / "result.json").read_text())
    assert "scheduled CR13 job" in result["error"]
    assert not result["full_pipeline_npu_acceptance"]
