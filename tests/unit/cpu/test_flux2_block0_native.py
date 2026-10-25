import json

import pytest
import torch

from scripts.diagnostics import flux2_block0_native as diagnostic


def test_matching_original_endpoints_allow_identity():
    original = torch.tensor([1.25, -2.0])
    observed = [original.clone() for _ in range(10)]
    assert all(diagnostic.identity_checks(original.clone(), observed, original).values())


def test_signed_zero_difference_rejects_bitwise_identity():
    original = torch.tensor([0.0])
    changed = torch.tensor([-0.0])
    assert torch.equal(original, changed)
    observed = [changed.clone() for _ in range(10)]
    checks = diagnostic.identity_checks(changed, observed, original)
    assert not checks["native_matches_full"]
    assert not checks["observed_matches_full"]


def test_changed_observed_endpoint_rejects_attribution():
    original = torch.tensor([1.0])
    observed = [original.clone() for _ in range(10)]
    observed[0] = torch.nextafter(original, torch.tensor([2.0]))
    checks = diagnostic.identity_checks(original, observed, original)
    assert checks["native_matches_full"]
    assert not checks["observed_matches_native"]
    assert not checks["observed_matches_full"]


def test_missing_observed_output_rejects_abi():
    value = torch.ones(1)
    with pytest.raises(ValueError, match="ABI"):
        diagnostic.identity_checks(value, [value] * 9, value)


def test_frozen_bundle_verifies_without_hardware(monkeypatch, tmp_path):
    # Use the retained real bundle's schema without loading its tensors or runtimes.
    manifest = {
        "sdk": "0.12.1.dev339+g781e7d9f",
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "compiler_source_commit": "781e7d9f1230a2b448f3d9d17389be7831a6975a",
        "primitive_commit": "017b55f8f23ba8b7c35c854cb584e948b874880b",
        "target": "RBLN-CR13",
        "num_devices": 1,
        "batch": 1,
        "sequence_length": 512,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "block_index": 0,
        "probe_parameters": 100930816,
        "source_model_parameters": 4022468096,
        "source_decoder_layers": 36,
        "partial_block_probe": True,
        "diagnostic_only": True,
        "full_pipeline_npu_acceptance": False,
        "original_graph_causal_attribution": False,
        "feature_names": diagnostic.FEATURES,
        "cases": [
            {"name": "short_component", "source_ci": 852},
            {"name": "native_valid512", "source_ci": 851},
        ],
        "files": {},
    }
    for name in diagnostic.FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"immutable fixture")
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": diagnostic.sha256(path)}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(diagnostic, "MANIFEST_SHA", diagnostic.sha256(path))
    assert diagnostic.verify_bundle(tmp_path) == manifest
    # A matching file hash cannot turn a partial probe into full-model acceptance.
    manifest["full_pipeline_npu_acceptance"] = True
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(diagnostic, "MANIFEST_SHA", diagnostic.sha256(path))
    with pytest.raises(ValueError, match="contract differs"):
        diagnostic.verify_bundle(tmp_path)
