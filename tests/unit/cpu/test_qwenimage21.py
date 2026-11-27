import unittest
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("has_artifacts", [False, True])
def test_pipeline_artifact_reuse_needs_no_source_checkpoint(monkeypatch, tmp_path, has_artifacts):
    from tests import test_diffusers

    model_cls = test_diffusers.TestQwenImage21Model
    source = tmp_path / "missing-source"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REUSE_ARTIFACTS_PATH", str(artifacts))
    monkeypatch.setenv("OPTIMUM_RBLN_TEST_LEVEL", "default")
    monkeypatch.setattr(model_cls, "HF_MODEL_ID", str(source))
    build = Mock(side_effect=AssertionError("Artifact reuse must not generate a source checkpoint"))
    monkeypatch.setattr(test_diffusers, "_save_tiny_qwenimage21_pipeline", build)

    if not has_artifacts:
        with pytest.raises(unittest.SkipTest, match="Compiled model not found"):
            model_cls.setUpClass()
    else:
        compiled = artifacts / model_cls.get_rbln_local_dir()
        compiled.mkdir()
        model = object()
        load = Mock(return_value=model)
        monkeypatch.setattr(model_cls.RBLN_CLASS, "from_pretrained", load)
        model_cls.setUpClass()
        try:
            assert model_cls.model is model
            load.assert_called_once_with(str(compiled))
        finally:
            model_cls.tearDownClass()
        assert not source.exists()

    build.assert_not_called()
