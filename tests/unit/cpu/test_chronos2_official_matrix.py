"""Reject a different SDK before any official-matrix device execution."""

import importlib.metadata
from types import SimpleNamespace

import pytest

from scripts.diagnostics.chronos2_official_matrix import official_sdk_receipt


def test_official_matrix_rejects_other_version_before_artifact_or_device_work(monkeypatch):
    distribution = importlib.metadata.distribution
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: (
            SimpleNamespace(version="0.12.1.dev329+g66a3078e") if name == "rebel-compiler" else distribution(name)
        ),
    )
    with pytest.raises(ValueError, match="Official SDK version differs"):
        official_sdk_receipt()
