import json

import pytest
import torch

from scripts.diagnostics import flux2_text_owner as diagnostic


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    files = {}
    hashes = {}
    for name in ("baseline", "selected4", "static_rope"):
        path = tmp_path / name / "compiled_model.rbln"
        path.parent.mkdir()
        path.write_bytes(name.encode())
        hashes[name] = diagnostic.sha256(path)
        files[str(path.relative_to(tmp_path))] = {"bytes": path.stat().st_size, "sha256": hashes[name]}
    manifest = {
        "sdk": diagnostic.SDK,
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "target": "RBLN-CR13",
        "batch": 1,
        "sequence_length": 512,
        "parameters": 4022468096,
        "decoder_layers": 36,
        "baseline_indices": list(range(37)),
        "selected_indices": [9, 18, 27, 36],
        "static_indices": list(range(37)),
        "required_indices": [9, 18, 27],
        "diagnostic_only": True,
        "full_pipeline_npu_acceptance": False,
        "public_selected4_ABI_compatible": False,
        "baseline_binary_sha256": hashes["baseline"],
        "selected_binary_sha256": hashes["selected4"],
        "static_binary_sha256": hashes["static_rope"],
        "files": files,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(diagnostic, "MANIFEST_SHA", diagnostic.sha256(path))
    return tmp_path, manifest


def test_owner_bundle_verifies_without_claiming_model_acceptance(bundle):
    root, _ = bundle
    assert diagnostic.main(["--verify-only", "--artifacts", str(root), "--output", str(root.parent / "output")]) == 0
    result = json.loads((root.parent / "output/result.json").read_text())
    assert not result["full_pipeline_npu_acceptance"]
    assert not result["diagnostic_completed"]


def test_owner_bundle_rejects_changed_graph(bundle):
    root, _ = bundle
    (root / "baseline/compiled_model.rbln").write_bytes(b"changed!")
    with pytest.raises(ValueError, match="checksum differs"):
        diagnostic.verify_bundle(root)


def test_owner_bundle_rejects_layer_removal_even_if_manifest_is_rehashed(bundle, monkeypatch):
    root, manifest = bundle
    manifest["decoder_layers"] = 27
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(diagnostic, "MANIFEST_SHA", diagnostic.sha256(path))
    with pytest.raises(ValueError, match="full-model text diagnostic contract"):
        diagnostic.verify_bundle(root)


def test_short_identity_requires_every_original_selected_value():
    values = {i: torch.tensor([float(i)]) for i in (9, 18, 27)}
    previous = {f"text_layer_{i}": value.clone() for i, value in values.items()}
    assert diagnostic.previous_identity("short_component", values, previous)
    values[27] = values[27] + 1
    assert not diagnostic.previous_identity("short_component", values, previous)


def test_full_prompt_identity_preserves_native_feature_channel_order():
    values = {i: torch.full((1, 512, 2560), float(i)) for i in (9, 18, 27)}
    previous = {"full_prompt_embeddings": torch.cat([values[i] for i in (9, 18, 27)], dim=-1)}
    assert diagnostic.previous_identity("full_original_prompt", values, previous)
    previous["full_prompt_embeddings"] = torch.cat([values[i] for i in (27, 18, 9)], dim=-1)
    assert not diagnostic.previous_identity("full_original_prompt", values, previous)


def test_execute_rejects_unscheduled_invocation_before_loading_graphs(bundle, monkeypatch):
    root, _ = bundle
    monkeypatch.delenv("BUILDKITE_BUILD_ID", raising=False)
    output = root.parent / "unscheduled-output"
    assert diagnostic.main(["--execute", "--artifacts", str(root), "--output", str(output)]) == 1
    result = json.loads((output / "result.json").read_text())
    assert "scheduled CR13 job" in result["error"]
    assert not result["full_pipeline_npu_acceptance"]


def test_owner_bundle_rejects_external_symlink(bundle):
    root, _ = bundle
    source = root / "baseline/compiled_model.rbln"
    outside = root.parent / "outside-owner.rbln"
    source.rename(outside)
    source.symlink_to(outside)
    with pytest.raises(ValueError, match="ordinary files"):
        diagnostic.verify_bundle(root)
