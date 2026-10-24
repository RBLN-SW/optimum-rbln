import json

import pytest
import torch

from scripts.diagnostics import flux2_transformer_frozen as diagnostic


def test_original_component_thresholds_reject_material_error():
    expected = torch.linspace(-2, 2, 128)
    assert diagnostic.metrics(expected.clone(), expected)["passed"]
    assert not diagnostic.metrics(-expected, expected)["passed"]
    actual = expected.clone()
    actual[3] = float("nan")
    assert not diagnostic.metrics(actual, expected)["passed"]
    json.dumps(diagnostic.metrics(actual, expected), allow_nan=False)


@pytest.mark.parametrize("environment", [{}, {"BUILDKITE_BUILD_ID": "test", "RBLN_TEST_NPU": "RBLN-CA25"}])
def test_unscheduled_or_wrong_target_execution_stops_before_device_import(monkeypatch, environment):
    monkeypatch.delenv("BUILDKITE_BUILD_ID", raising=False)
    monkeypatch.delenv("RBLN_TEST_NPU", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="scheduled CR13"):
        diagnostic.require_scheduled_execution()


def test_artifact_compatibility_cannot_substitute_numerical_execution(monkeypatch):
    monkeypatch.setenv("BUILDKITE_BUILD_ID", "scheduled")
    monkeypatch.setenv("RBLN_TEST_NPU", "RBLN-CR13")
    monkeypatch.setenv("REUSE_ARTIFACTS_PATH", "compat")
    with pytest.raises(ValueError, match="compatibility"):
        diagnostic.require_scheduled_execution()
