"""Supplement pooled full-model parity with immutable per-series evidence."""

from types import SimpleNamespace

import pytest
import torch

from scripts.diagnostics.chronos2_prediction_audit import PredictionAudit, input_identity, series_metrics


def test_low_energy_failed_series_cannot_hide_behind_high_energy_member():
    reference = torch.linspace(-1, 1, 24).reshape(2, 3, 4)
    reference[1] *= 1000
    actual = reference.clone()
    actual[0] *= 1.2
    pooled = (actual - reference).square().mean().sqrt() / reference.square().mean().sqrt()
    assert pooled < 0.05
    rows = series_metrics(actual, reference)
    assert not rows[0]["passed"]
    assert rows[1]["passed"]


def test_zero_variance_requires_exact_equality_and_nonfinite_values_fail():
    expected = torch.zeros(1, 2, 3)
    assert series_metrics(expected, expected)[0]["passed"]
    assert not series_metrics(torch.full_like(expected, 1e-10), expected)[0]["passed"]
    assert not series_metrics(torch.full_like(expected, torch.nan), expected)[0]["passed"]


def test_input_binding_preserves_nan_bytes_and_distinguishes_changed_requests():
    context = torch.tensor([[1.0, torch.nan, 3.0]])
    a = {"context": context, "num_output_patches": 2}
    assert input_identity(a) == input_identity({"num_output_patches": 2, "context": context.clone()})
    changed = context.clone()
    changed[0, 0] += 1
    assert input_identity(a)[0] != input_identity({**a, "context": changed})[0]


def test_saved_oracle_and_outputs_bind_repeats_and_expose_changed_results(tmp_path):
    model = torch.nn.Module()
    audit = PredictionAudit(tmp_path / "predictions", {})
    inputs = {"context": torch.ones(2, 4), "num_output_patches": 2}
    expected = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    audit.reference(model, inputs, SimpleNamespace(quantile_preds=expected))
    for actual in [expected.clone(), expected.clone(), expected + 1]:
        with audit.product_request(inputs):
            audit.product_output(SimpleNamespace(quantile_preds=actual))
    report = audit.finish()
    assert report["matched_calls"] == 3
    assert report["unique_reference_inputs"] == 1
    assert not report["identical_input_outputs_exact"]
    assert not report["all_series_passed"]
    for name, row in report["files"].items():
        assert (audit.root / name).stat().st_size == row["bytes"]
    first = report["calls"][0]
    saved = torch.load(audit.root / first["reference_file"], weights_only=True)["prediction"]
    torch.testing.assert_close(saved, expected, rtol=0, atol=0)


def test_product_request_mutation_is_rejected_before_evidence_is_accepted(tmp_path):
    audit = PredictionAudit(tmp_path / "predictions", {})
    inputs = {"context": torch.ones(1, 4)}
    with pytest.raises(ValueError, match="modified original request"), audit.product_request(inputs):
        inputs["context"][0, 0] = 2
