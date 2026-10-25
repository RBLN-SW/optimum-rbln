"""The compiler candidate uses the original numeric criteria, never the old BAD hash."""

import torch

from scripts.diagnostics.chronos2_encoder0_candidate_run import boundary_metrics


def test_candidate_boundary_metrics_reject_large_scale_error_and_permutation():
    reference = torch.linspace(-1, 1, 1024)
    assert boundary_metrics(reference * 1.001, reference)["passed"]
    assert not boundary_metrics(reference * 1.1, reference)["passed"]
    assert not boundary_metrics(reference.flip(0), reference)["passed"]


def test_zero_energy_reference_requires_exact_zero():
    reference = torch.zeros(16)
    assert boundary_metrics(reference, reference)["passed"]
    assert not boundary_metrics(torch.full_like(reference, 1e-10), reference)["passed"]
