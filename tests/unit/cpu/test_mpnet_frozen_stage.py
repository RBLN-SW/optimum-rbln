import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.diagnostics import mpnet_b2048_stage as staging
from tests.unit.cpu.test_mpnet_frozen_harness import artifacts as make_artifacts


def artifacts(tmp_path, monkeypatch):
    source = make_artifacts(tmp_path, monkeypatch)
    monkeypatch.setattr(
        staging, "MANIFEST_SHA256", hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    )
    return source


def test_stage_import_uses_only_standard_library():
    code = """
import sys
class RejectLearnedStack:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'rebel', 'optimum', 'torch', 'transformers', 'pytest'}:
            raise AssertionError('Stage imported the learned/compiler stack: ' + name)
sys.meta_path.insert(0, RejectLearnedStack())
from scripts.diagnostics import mpnet_b2048_stage
assert mpnet_b2048_stage.harness.SDK == '0.12.1.dev329+g66a3078e'
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_atomic_hash_verified_copy_and_identical_reuse(tmp_path, monkeypatch):
    source = artifacts(tmp_path, monkeypatch)
    destination = tmp_path / "cross" / "frozen"
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    assert staging.stage(source, destination)["state"] == "staged"
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == original
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original
    monkeypatch.setattr(staging.shutil, "copy2", lambda *args: pytest.fail("identical target must not be copied"))
    assert staging.stage(source, destination)["state"] == "reused"
    assert sorted(p.name for p in destination.parent.iterdir()) == ["frozen"]


def test_corrupt_source_is_rejected_before_copy(tmp_path, monkeypatch):
    source = artifacts(tmp_path, monkeypatch)
    (source / "compiled_model.rbln").write_text("corrupt")
    destination = tmp_path / "cross" / "frozen"
    with pytest.raises(ValueError, match="Checksum"):
        staging.stage(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("conflict", ["empty", "corrupt", "different-manifest"])
def test_existing_conflict_is_never_overwritten(tmp_path, monkeypatch, conflict):
    source = artifacts(tmp_path, monkeypatch)
    destination = tmp_path / "cross" / "frozen"
    destination.parent.mkdir()
    if conflict == "empty":
        destination.mkdir()
    else:
        shutil.copytree(source, destination)
        if conflict == "corrupt":
            (destination / "compiled_model.rbln").write_text("corrupt")
        else:
            manifest = json.loads((destination / "manifest.json").read_text())
            manifest["compile_checkout_head"] = "different receipt"
            (destination / "manifest.json").write_text(json.dumps(manifest))
    original = {p.name: p.read_bytes() for p in destination.iterdir()}
    message = {"empty": "exactly five", "corrupt": "Checksum", "different-manifest": "manifest differs"}
    with pytest.raises(ValueError, match=message[conflict]):
        staging.stage(source, destination)
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == original


@pytest.mark.parametrize("failure", ["interrupt", "copied-corruption"])
def test_partial_copy_is_never_published_and_temporary_files_are_removed(tmp_path, monkeypatch, failure):
    source = artifacts(tmp_path, monkeypatch)
    destination = tmp_path / "cross" / "frozen"
    copy = staging.shutil.copy2

    def faulty_copy(src, dst):
        if failure == "interrupt":
            raise OSError("interrupted copy")
        result = copy(src, dst)
        if Path(src).name == "compiled_model.rbln":
            Path(dst).write_text("corrupted during copy")
        return result

    monkeypatch.setattr(staging.shutil, "copy2", faulty_copy)
    with pytest.raises((OSError, ValueError)):
        staging.stage(source, destination)
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []


def test_publication_is_one_verified_sibling_directory_rename(tmp_path, monkeypatch):
    source = artifacts(tmp_path, monkeypatch)
    destination = tmp_path / "cross" / "frozen"
    rename = Path.rename
    publications = []

    def checked_rename(temporary, target):
        assert target == destination
        assert temporary.parent == destination.parent
        assert not destination.exists()
        assert staging.stage_lock(destination).is_dir()
        staging.verify_bundle(temporary)
        publications.append((temporary, target))
        return rename(temporary, target)

    monkeypatch.setattr(Path, "rename", checked_rename)
    assert staging.stage(source, destination)["state"] == "staged"
    assert len(publications) == 1
    assert not staging.stage_lock(destination).exists()


def test_busy_cooperative_writer_lock_is_not_removed_or_bypassed(tmp_path, monkeypatch):
    source = artifacts(tmp_path, monkeypatch)
    destination = tmp_path / "cross" / "frozen"
    lock = staging.stage_lock(destination)
    lock.mkdir(parents=True)
    monkeypatch.setattr(staging.shutil, "copy2", lambda *args: pytest.fail("busy writer must not copy"))
    with pytest.raises(FileExistsError, match="another staging writer"):
        staging.stage(source, destination)
    assert lock.is_dir()
    assert not destination.exists()


def test_concurrent_identical_publish_is_reused(tmp_path, monkeypatch):
    source = artifacts(tmp_path, monkeypatch)
    destination = tmp_path / "cross" / "frozen"
    copy = staging.shutil.copy2
    peer_published = False

    def race(src, dst):
        nonlocal peer_published
        if not peer_published:
            shutil.copytree(source, destination, copy_function=copy)
            peer_published = True
        return copy(src, dst)

    monkeypatch.setattr(staging.shutil, "copy2", race)
    assert staging.stage(source, destination)["state"] == "reused"
    assert sorted(p.name for p in destination.parent.iterdir()) == ["frozen"]


def test_manifest_symlink_is_rejected(tmp_path, monkeypatch):
    source = artifacts(tmp_path, monkeypatch)
    external = tmp_path / "external-manifest.json"
    (source / "manifest.json").rename(external)
    (source / "manifest.json").symlink_to(external)
    with pytest.raises(ValueError, match="ordinary"):
        staging.stage(source, tmp_path / "destination")


def test_proven_queue_route_resources_sdk_and_original_quality_gates():
    pipeline = yaml.safe_load(Path(".buildkite/pr.yml").read_text())
    steps = pipeline["steps"]
    assert [s["label"] for s in steps[:3]] == [
        ":closed_lock_with_key: team-member gate",
        ":mag: code quality",
        ":page_facing_up: docstrings",
    ]
    assert steps[3] == steps[5] == {"wait": None}
    stage, consumer = steps[4], steps[6]
    assert stage["agents"] == {"queue": "k8s"}
    assert "resources" not in stage
    assert stage["command"] == ["python3 -m scripts.diagnostics.mpnet_b2048_stage"]
    assert consumer["agents"] == {"queue": "udc"}
    assert consumer["resources"] == {
        "cpu": {"requests": "16"},
        "memory": {"requests": "128Gi", "limits": "128Gi"},
        "npu": {"count": 1, "product": "RBLN-CR13"},
    }
    assert consumer["env"]["RBLN_TEST_NPU"] == "RBLN-CR13"
    assert str(staging.DESTINATION) in consumer["command"][-1]
    assert str(staging.SOURCE) not in consumer["command"][-1]
    assert yaml.safe_load(Path(".github/version.yaml").read_text())["rebel_compiler_version"] == staging.harness.SDK
