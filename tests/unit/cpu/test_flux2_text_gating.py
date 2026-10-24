import json

import pytest
import torch

from scripts.diagnostics import flux2_text_gating as diagnostic


def test_one_tensor_is_one_output_even_when_its_leading_dimension_is_four():
    value = torch.zeros(4, 2)
    result = diagnostic.as_outputs(value, 1)
    assert len(result) == 1
    assert result[0] is value
    with pytest.raises(ValueError, match="output count"):
        diagnostic.as_outputs(value, 4)


def test_observed_gate_requires_all_four_outputs():
    value = torch.zeros(1)
    assert len(diagnostic.as_outputs((value, value, value, value), 4)) == 4
    with pytest.raises(ValueError, match="output count"):
        diagnostic.as_outputs((value, value, value), 4)


def test_unscheduled_gating_probe_rejected_before_loading_graphs(tmp_path, monkeypatch):
    monkeypatch.delenv("BUILDKITE_BUILD_ID", raising=False)
    assert diagnostic.main(["--execute", "--artifacts", str(tmp_path / "missing"), "--output", str(tmp_path)]) == 1
    result = json.loads((tmp_path / "result.json").read_text())
    assert "scheduled CR13 job" in result["error"]
    assert not result["full_pipeline_npu_acceptance"]
    assert not result["original_graph_causal_attribution"]
