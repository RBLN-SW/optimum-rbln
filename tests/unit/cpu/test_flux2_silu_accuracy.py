import json

import pytest

from scripts.diagnostics import flux2_silu_accuracy as diagnostic


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    manifest = {
        "sdk": diagnostic.SDK,
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "target": "RBLN-CR13",
        "shape": [1, 512, 9728],
        "variants": diagnostic.VARIANTS,
        "probe_only": True,
        "not_adopted": True,
        "full_pipeline_npu_acceptance": False,
        "original_graph_causal_attribution": False,
        "implementation_candidates": ["stable_exp"],
        "placement_audit_pending": False,
        "accuracy_target_nrmse": 0.005,
        "accuracy_target_pcc": 0.9999,
        "cases": [{"name": name} for name in diagnostic.CASES],
        "files": {},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(diagnostic, "MANIFEST_SHA", diagnostic.sha256(path))
    return tmp_path, manifest


def rehash(root, manifest, monkeypatch):
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(diagnostic, "MANIFEST_SHA", diagnostic.sha256(path))


def test_verification_does_not_adopt_candidate_or_claim_model_support(bundle):
    root, _ = bundle
    output = root.parent / "verify-result"
    assert diagnostic.main(["--verify-only", "--artifacts", str(root), "--output", str(output)]) == 0
    result = json.loads((output / "result.json").read_text())
    assert not result["full_pipeline_npu_acceptance"]
    assert not result["original_graph_causal_attribution"]
    assert result["not_adopted"]
    assert not result["diagnostic_completed"]


def test_rehashed_manifest_cannot_relax_accuracy_target(bundle, monkeypatch):
    root, manifest = bundle
    manifest["accuracy_target_nrmse"] = 0.05
    rehash(root, manifest, monkeypatch)
    with pytest.raises(ValueError, match="experiment contract"):
        diagnostic.verify_bundle(root)


def test_rehashed_manifest_cannot_claim_original_model_acceptance(bundle, monkeypatch):
    root, manifest = bundle
    manifest["full_pipeline_npu_acceptance"] = True
    rehash(root, manifest, monkeypatch)
    with pytest.raises(ValueError, match="experiment contract"):
        diagnostic.verify_bundle(root)


def test_rehashed_manifest_cannot_run_zero_cases(bundle, monkeypatch):
    root, manifest = bundle
    manifest["cases"] = []
    rehash(root, manifest, monkeypatch)
    with pytest.raises(ValueError, match="Required actual native inputs"):
        diagnostic.verify_bundle(root)


def test_unscheduled_execution_rejected_before_artifact_access(tmp_path, monkeypatch):
    monkeypatch.delenv("BUILDKITE_BUILD_ID", raising=False)
    assert diagnostic.main(["--execute", "--artifacts", str(tmp_path / "missing"), "--output", str(tmp_path)]) == 1
    result = json.loads((tmp_path / "result.json").read_text())
    assert "scheduled CR13 job" in result["error"]
    assert not result["full_pipeline_npu_acceptance"]
