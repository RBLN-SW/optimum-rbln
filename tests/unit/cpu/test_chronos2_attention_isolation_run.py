"""Reject invalid frozen operator bundles before any device execution."""

import hashlib
import json
import sys

import pytest

from scripts.diagnostics import chronos2_attention_isolation_run as runner
from scripts.diagnostics.chronos2_attention_isolation_stage import stage


def write_manifest(root, value):
    path = root / "manifest.json"
    path.write_text(json.dumps(value))
    return runner.file_sha(path)


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for name in runner.FILES:
        (root / name).write_bytes(b"frozen-operator-fixture")
    manifest = {
        "files": {
            name: {"bytes": (root / name).stat().st_size, "sha256": runner.file_sha(root / name)}
            for name in runner.FILES
        },
        "model": "amazon/chronos-2",
        "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
        "target": "RBLN-CA25",
        "full_model_acceptance": False,
        "original_device_attribution": False,
        "original_identity_gate_failed": True,
        "observation_rows": [0, 1, 2, 3, 4, 511],
        "cases": [
            {"name": "sdpa_b1", "batch": 1, "explicit": False},
            {"name": "sdpa_b6", "batch": 6, "explicit": False},
            {"name": "explicit_b6", "batch": 6, "explicit": True},
        ],
        "criteria": {"nrmse_max": 0.05, "pcc_min": 0.99, "repeat_count": 10},
    }
    return root, manifest, write_manifest(root, manifest)


def test_valid_integrity_and_staging_preserve_membership(bundle, tmp_path):
    root, manifest, digest = bundle
    assert runner.verify_bundle(root, digest) == manifest
    destination = tmp_path / "staged"
    assert stage(root, destination, digest) == "staged_verified"
    assert stage(root, destination, digest) == "verified_existing"
    assert runner.verify_bundle(destination, digest) == manifest


def test_same_size_corruption_is_rejected(bundle):
    root, _, digest = bundle
    file = root / "inputs.pt"
    file.write_bytes(b"X" * file.stat().st_size)
    with pytest.raises(ValueError, match="size/SHA mismatch"):
        runner.verify_bundle(root, digest)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("scope", "Invalid model"),
        ("rows", "Captured row mapping"),
        ("geometry", "Operator case geometry"),
        ("criteria", "Diagnostic criteria"),
    ],
)
def test_reject_scope_mapping_geometry_and_criteria_changes(bundle, change, message):
    root, manifest, _ = bundle
    if change == "scope":
        manifest["original_device_attribution"] = True
    elif change == "rows":
        manifest["observation_rows"] = [1, 0, 2, 3, 4, 511]
    elif change == "geometry":
        manifest["cases"][1]["batch"] = 3
    else:
        manifest["criteria"]["nrmse_max"] = 0.1
    digest = write_manifest(root, manifest)
    with pytest.raises(ValueError, match=message):
        runner.verify_bundle(root, digest)


def test_unreserved_execution_is_rejected_before_tensor_loading(bundle, monkeypatch):
    root, _, digest = bundle
    monkeypatch.delenv("BUILDKITE_JOB_ID", raising=False)
    monkeypatch.delenv("RBLN_TEST_NPU", raising=False)
    monkeypatch.setattr(sys, "argv", ["probe", "--bundle", str(root), "--manifest-sha256", digest])
    with pytest.raises(ValueError, match="reserved scheduler"):
        runner.main()


def test_manifest_digest_is_required(bundle):
    root, _, _ = bundle
    with pytest.raises(ValueError, match="Manifest SHA"):
        runner.verify_bundle(root, hashlib.sha256(b"wrong manifest").hexdigest())


def test_existing_staging_is_never_overwritten(bundle, tmp_path):
    root, _, digest = bundle
    destination = tmp_path / "staged"
    stage(root, destination, digest)
    bad = destination / "goldens.pt"
    bad.write_bytes(b"different frozen evidence")
    retained = bad.read_bytes()
    with pytest.raises(ValueError, match="size/SHA mismatch"):
        stage(root, destination, digest)
    assert bad.read_bytes() == retained


def test_symlink_cannot_replace_frozen_file(bundle, tmp_path):
    root, _, digest = bundle
    file = root / "inputs.pt"
    external = tmp_path / "outside"
    external.write_bytes(file.read_bytes())
    file.unlink()
    file.symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        runner.verify_bundle(root, digest)
