import hashlib
import json

import pytest

from scripts.diagnostics import flux2_pipeline_frozen as diagnostic


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    artifact = tmp_path / "model" / "compiled.rbln"
    artifact.parent.mkdir()
    artifact.write_bytes(b"immutable trained binary")
    manifest = {
        "revision": diagnostic.REVISION,
        "sdk": diagnostic.SDK,
        "target": "RBLN-CR13",
        "num_devices": 1,
        "batch": 1,
        "image_size": [1024, 1024],
        "prompt_length": 512,
        "reference_image_count": 0,
        "denoising_steps": 4,
        "guidance": 1.0,
        "hybrid_source_overlay": True,
        "official_patched_SDK_acceptance": False,
        "compiler_source_commit": "78fd8c151d9882fdd00476c709ffcfb9bbf39807",
        "unchanged_goldens_thresholds_and_other_components": True,
        "source_overlay": {"method_calls": 36, "source_dtypes": {"float32": 36}},
        "files": {"model/compiled.rbln": {"bytes": artifact.stat().st_size, "sha256": diagnostic.sha256(artifact)}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(diagnostic, "MANIFEST_SHA", diagnostic.sha256(path))
    return tmp_path, manifest


def test_frozen_native_profile_verifies_without_runtime(frozen, monkeypatch):
    root, manifest = frozen
    assert diagnostic.verify_bundle(root) == manifest

    def reject():
        raise AssertionError("Verify-only must not enter hardware execution")

    monkeypatch.setattr(diagnostic, "require_scheduled_execution", reject)
    assert diagnostic.main(["--verify-only", "--artifacts", str(root), "--output", str(root.parent / "results")]) == 0


def test_frozen_rejects_modified_binary(frozen):
    root, _ = frozen
    (root / "model/compiled.rbln").write_bytes(b"modified trained binary!")
    with pytest.raises(ValueError, match="content differs"):
        diagnostic.verify_bundle(root)


def test_frozen_rejects_added_files(frozen):
    root, _ = frozen
    (root / "extra.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="unexpected files"):
        diagnostic.verify_bundle(root)


def test_frozen_rejects_symlink_even_when_content_matches(frozen):
    root, _ = frozen
    source = root / "model/compiled.rbln"
    outside = root.parent / "outside.rbln"
    source.rename(outside)
    source.symlink_to(outside)
    with pytest.raises(ValueError, match="ordinary files"):
        diagnostic.verify_bundle(root)


def test_frozen_rejects_shortened_native_context(frozen, monkeypatch):
    root, manifest = frozen
    manifest["prompt_length"] = 128
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(diagnostic, "MANIFEST_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="profile differs"):
        diagnostic.verify_bundle(root)


def test_frozen_rejects_incomplete_source_conversion(frozen, monkeypatch):
    root, manifest = frozen
    manifest["source_overlay"]["method_calls"] = 35
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(diagnostic, "MANIFEST_SHA", diagnostic.sha256(path))
    with pytest.raises(ValueError, match="36 FP32 SiLU"):
        diagnostic.verify_bundle(root)


def test_frozen_rejects_official_wheel_claim(frozen, monkeypatch):
    root, manifest = frozen
    manifest["official_patched_SDK_acceptance"] = True
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(diagnostic, "MANIFEST_SHA", diagnostic.sha256(path))
    with pytest.raises(ValueError, match="profile differs"):
        diagnostic.verify_bundle(root)
