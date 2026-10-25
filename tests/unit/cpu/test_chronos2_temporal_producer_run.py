"""Frozen-bundle checks run without model imports or physical devices."""

import hashlib
import json
import sys

import pytest

from scripts.diagnostics import chronos2_temporal_producer_run as runner


def write_bundle(root, **overrides):
    root.mkdir()
    names = (
        "encoder0_producers.rbln",
        "exact-ci819-inputs.pt",
        "cpu-boundary-references.pt",
        "cpu-producer-references.pt",
        "original-inputs.json",
        "trained-parameters.json",
        "cpu-probe-receipt.json",
        "compile-result.json",
    )
    files = {}
    for name in names:
        value = ("fixture-" + name).encode()
        (root / name).write_bytes(value)
        files[name] = {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    manifest = {
        "files": files,
        "full_model_acceptance": False,
        "target": "RBLN-CA25",
        "model_input_geometry": [512, 577, 768],
        "observation_row_indices": [0, 1, 2, 3, 4, 511],
    }
    manifest.update(overrides)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()


def test_verify_bundle_without_device_imports(tmp_path):
    root = tmp_path / "bundle"
    digest = write_bundle(root)
    manifest = runner.verify_bundle(root, digest)
    assert len(manifest["files"]) == 8
    assert not any(name == "rebel" or name.startswith(("rebel.", "tvm.")) for name in sys.modules)


def test_same_size_corruption_rejected(tmp_path):
    root = tmp_path / "bundle"
    digest = write_bundle(root)
    path = root / "encoder0_producers.rbln"
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ValueError, match="Frozen file size/SHA mismatch"):
        runner.verify_bundle(root, digest)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_input_geometry": [3, 577, 768]},
        {"observation_row_indices": [0, 1, 2]},
    ],
)
def test_geometry_or_observation_mapping_rejected(tmp_path, overrides):
    root = tmp_path / "bundle"
    digest = write_bundle(root, **overrides)
    with pytest.raises(ValueError, match=r"geometry changed|row mapping changed"):
        runner.verify_bundle(root, digest)


def test_full_execution_denied_before_device_imports(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    digest = write_bundle(root)
    output = tmp_path / "must-not-exist"
    monkeypatch.delenv("BUILDKITE_JOB_ID", raising=False)
    monkeypatch.delenv("RBLN_TEST_NPU", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["producer-probe", "--bundle", str(root), "--manifest-sha256", digest, "--out", str(output)],
    )
    with pytest.raises(ValueError, match="reserved scheduler CA25"):
        runner.main()
    assert not output.exists()
    assert not any(name == "rebel" or name.startswith(("rebel.", "tvm.")) for name in sys.modules)
