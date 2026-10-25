import pytest

from scripts.diagnostics import flux2_softmax_accuracy as diagnostic


def test_blocked_mapping_preserves_key_lanes_and_query_rows():
    import torch

    values = torch.arange(768, dtype=torch.float32).reshape(1, 2, 3, 128)
    values[0, 0, 0, 0] = -0.0
    blocked = diagnostic.to_blocked(values)
    assert blocked.shape == (1, 2, 2, 3, 64)
    assert blocked[0, 1, 1, 2, 5] == values[0, 1, 2, 69]
    restored = diagnostic.from_blocked(blocked)
    assert torch.equal(restored.contiguous().view(torch.uint8), values.contiguous().view(torch.uint8))


def test_non_aligned_keys_are_not_silently_padded():
    import torch

    with pytest.raises(ValueError, match="64-lane"):
        diagnostic.to_blocked(torch.zeros(1, 2, 3, 63))


def test_exact_rounded_ideal_has_no_implementation_difference():
    import torch

    reference = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float32)
    ideal = torch.tensor([0.1001, 0.2001, 0.4001], dtype=torch.float32)
    result = diagnostic.quantization_metrics(ideal, reference, ideal)
    assert result["actual_vs_cpu"]["nrmse"] > 0
    assert result["actual_vs_rounded_ideal"]["nrmse"] == 0
    assert result["actual_vs_cpu"] == result["rounded_ideal_vs_cpu"]


def test_metrics_reject_shape_drift_before_flattening():
    import torch

    reference = torch.arange(6, dtype=torch.float32).reshape(2, 3) + 1
    with pytest.raises(ValueError, match="output ABI"):
        diagnostic.quantization_metrics(reference.reshape(3, 2), reference, reference)
