"""Scheduler guards, exact original selection and reload lifetime without an NPU."""

from contextlib import nullcontext
from pathlib import Path

import pytest

from scripts.diagnostics.mpnet_shared_bias_native512 import (
    CLASSES,
    fresh_reload_original_parity,
    guard_environment,
    select_original_classes,
)


@pytest.mark.parametrize("name", ["SAVE_ARTIFACTS_PATH", "REUSE_ARTIFACTS_PATH"])
@pytest.mark.parametrize("value", ["", "artifact-directory"])
def test_artifact_compatibility_modes_are_rejected(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        guard_environment(verify_only=True)


def test_local_runtime_path_requires_scheduler(monkeypatch):
    for name in ("SAVE_ARTIFACTS_PATH", "REUSE_ARTIFACTS_PATH", "BUILDKITE_BUILD_ID"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="BUILDKITE_BUILD_ID"):
        guard_environment(verify_only=False)
    guard_environment(verify_only=True)


def test_collection_is_exactly_original_native512_b1_b2_b4():
    classes, selection = select_original_classes()
    assert [cls.__name__ for cls in classes] == list(CLASSES)
    assert len(selection) == 27
    assert all("NativeMaxBatch" in name for name in selection)


def test_fresh_reload_releases_old_runtime_and_calls_original_parity(tmp_path):
    transitions = []
    events = []

    class Model:
        def __init__(self, label):
            self.label = label
            self.compiled_models = [object()]

        def save_pretrained(self, directory):
            (Path(directory) / "compiled_model.rbln").write_bytes(b"saved-candidate")
            transitions.append(("save", self.label))

        def __del__(self):
            transitions.append(("release", self.label))

    class Product:
        @staticmethod
        def from_pretrained(directory, **kwargs):
            assert transitions[-1] == ("release", "original")
            assert kwargs == {"export": False, "revision": "original-pinned-revision"}
            assert (Path(directory) / "compiled_model.rbln").read_bytes() == b"saved-candidate"
            transitions.append(("load", "fresh"))
            return Model("fresh")

    class NativeCase:
        DEVICE = 0
        HF_CONFIG_KWARGS = {"revision": "original-pinned-revision"}
        RBLN_CLASS = Product
        model = Model("original")

        def __init__(self, method):
            assert method == "test_embedding_parity"

        def test_embedding_parity(self):
            assert self.model.label == "fresh"
            transitions.append(("original_parity", "fresh"))

    fresh_reload_original_parity(NativeCase, tmp_path, events.append, context_factory=lambda **_: nullcontext())
    assert transitions == [
        ("save", "original"),
        ("release", "original"),
        ("load", "fresh"),
        ("original_parity", "fresh"),
        ("release", "fresh"),
    ]
    assert "model" not in NativeCase.__dict__
    assert events[0]["phase"] == "fresh_reload_original_embedding_parity"
    assert events[0]["passed"] is True
