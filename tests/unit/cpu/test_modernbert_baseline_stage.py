import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.diagnostics import modernbert_baseline_stage as staging


def source(tmp_path, monkeypatch):
    path = tmp_path / "nas" / staging.FILENAME
    path.parent.mkdir()
    path.write_bytes(b"frozen-native-8192-ci806-baseline")
    monkeypatch.setattr(staging, "BYTES", path.stat().st_size)
    monkeypatch.setattr(staging, "SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_staging_has_no_learned_stack_dependency():
    code = """
import sys
class RejectStack:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'torch', 'rebel', 'optimum', 'transformers'}:
            raise AssertionError('Unexpected learned stack import: ' + name)
sys.meta_path.insert(0, RejectStack())
from scripts.diagnostics import modernbert_baseline_stage
assert modernbert_baseline_stage.BYTES == 50465191
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_verified_atomic_publish_and_identical_reuse(tmp_path, monkeypatch):
    original = source(tmp_path, monkeypatch)
    destination = tmp_path / "cross" / "baseline" / staging.FILENAME
    rename = Path.rename
    publications = []

    def publication(temporary, target):
        assert target == destination.parent
        assert temporary.parent == target.parent
        assert not target.exists()
        staging.verify(temporary / staging.FILENAME)
        publications.append(target)
        return rename(temporary, target)

    monkeypatch.setattr(Path, "rename", publication)
    assert staging.stage(original, destination)["state"] == "staged"
    assert publications == [destination.parent]
    assert destination.read_bytes() == original.read_bytes()
    monkeypatch.setattr(shutil, "copy2", lambda *args: pytest.fail("Identical content must be reused"))
    assert staging.stage(original, destination)["state"] == "reused"


@pytest.mark.parametrize("conflict", ["empty", "unknown", "corrupt"])
def test_unknown_destination_is_never_overwritten(tmp_path, monkeypatch, conflict):
    original = source(tmp_path, monkeypatch)
    destination = tmp_path / "cross" / "baseline" / staging.FILENAME
    destination.parent.mkdir(parents=True)
    if conflict == "unknown":
        (destination.parent / "unrelated.txt").write_text("preserve")
    elif conflict == "corrupt":
        destination.write_bytes(b"bad")
    before = {p.name: p.read_bytes() for p in destination.parent.iterdir()}
    with pytest.raises(ValueError, match=r"unknown|ordinary"):
        staging.stage(original, destination)
    assert {p.name: p.read_bytes() for p in destination.parent.iterdir()} == before


@pytest.mark.parametrize("failure", ["interrupted", "corrupted-copy"])
def test_partial_copy_is_not_published_and_only_own_temporary_state_removed(tmp_path, monkeypatch, failure):
    original = source(tmp_path, monkeypatch)
    destination = tmp_path / "cross" / "baseline" / staging.FILENAME
    copy = shutil.copy2

    def faulty_copy(src, dst):
        if failure == "interrupted":
            raise OSError("interrupted")
        copy(src, dst)
        Path(dst).write_bytes(b"bad")

    monkeypatch.setattr(shutil, "copy2", faulty_copy)
    with pytest.raises((OSError, ValueError)):
        staging.stage(original, destination)
    assert not destination.exists()
    assert list(destination.parent.parent.iterdir()) == []


def test_busy_writer_lock_is_not_removed(tmp_path, monkeypatch):
    original = source(tmp_path, monkeypatch)
    destination = tmp_path / "cross" / "baseline" / staging.FILENAME
    lock = destination.parent.with_name(".baseline.stage-lock")
    lock.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Another"):
        staging.stage(original, destination)
    assert lock.is_dir()


def test_pipeline_stages_on_k8s_before_udc_consumer():
    pipeline = yaml.safe_load(Path(".buildkite/pr.yml").read_text())
    stage = next(s for s in pipeline["steps"] if s.get("key") == "modernbert-baseline-stage")
    consumer = next(s for s in pipeline["steps"] if "selected boundaries" in s.get("label", ""))
    assert stage["agents"]["queue"] == "k8s"
    assert consumer["agents"]["queue"] == "udc"
    assert consumer["depends_on"] == stage["key"]
    assert str(staging.DESTINATION) in consumer["command"][-1]
