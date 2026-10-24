import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.diagnostics import dinov2_native_b512_frozen as harness


def artifacts(tmp_path):
    root = tmp_path / "frozen"
    config = {
        "batch_size": 512,
        "image_size": 518,
        "encoder_chunk_size": 6,
        "_compile_cfgs": [
            {
                "compiled_model_name": name,
                "npu": "RBLN-CA25",
                "num_devices": 1,
                "input_info": [
                    *([["pixel_values", [512, 3, 518, 518], "float32"]] if name == "embedding" else []),
                    ["activation", [512, 1370, 768], "float32"],
                ],
            }
            for name in harness.STAGES
        ],
    }
    for kind in harness.CLASSES.values():
        directory = root / kind
        directory.mkdir(parents=True)
        (directory / "rbln_config.json").write_text(json.dumps(config))
        (directory / "config.json").write_text(json.dumps({"num_hidden_layers": 12}))
        for name in harness.STAGES:
            (directory / f"{name}.rbln").write_bytes(name.encode())
    manifest = {"source_compiler": harness.COMPILER, "source_optimum": harness.OPTIMUM, "files": {}}
    for path in root.rglob("*"):
        if path.is_file():
            manifest["files"][str(path.relative_to(root))] = {
                "bytes": path.stat().st_size,
                "sha256": harness.sha256(path),
            }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def inspect(path):
    return {"uuid": path.parent.name, "npu": "RBLN-CA25"}


def test_manifest_all_files_sources_config_and_shared_sessions(tmp_path):
    root = artifacts(tmp_path)
    receipt = harness.verify_artifacts(root, inspect_model=inspect)
    assert set(receipt["metadata"]) == {"encoder", "classifier"}
    with pytest.raises(ValueError, match="source heads"):
        harness.verify_artifacts(root, expected_compiler="wrong", inspect_model=inspect)
    with pytest.raises(ValueError, match="compile session"):
        harness.verify_artifacts(root, inspect_model=lambda path: {"uuid": str(path), "npu": "RBLN-CA25"})
    (root / "encoder" / "embedding.rbln").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        harness.verify_artifacts(root, inspect_model=inspect)


def test_manifest_requires_unlisted_files(tmp_path):
    root = artifacts(tmp_path)
    (root / "encoder" / "extra").write_text("unexpected")
    with pytest.raises(ValueError, match="every"):
        harness.verify_artifacts(root, inspect_model=inspect)


def test_verify_only_never_runs_tests_and_writes_receipts(tmp_path, monkeypatch):
    root = artifacts(tmp_path)
    original = harness.verify_artifacts
    monkeypatch.setattr(harness, "verify_artifacts", lambda *args: original(*args, inspect_model=inspect))
    monkeypatch.setattr(harness, "run_tests", lambda *args: pytest.fail("runtime execution"))
    output = tmp_path / "output"
    assert harness.main(["--artifacts", str(root), "--output", str(output), "--verify-only"]) == 0
    result = json.loads((output / "result.json").read_text())
    assert result["phase"] == "verified_only"
    assert result["runtime_executed"] is False
    assert result["accepted"] is False
    assert (output / "junit.xml").exists()


def test_preflight_error_always_reports_and_blocks_skip_environment(tmp_path, monkeypatch):
    output = tmp_path / "output"
    monkeypatch.setenv("REUSE_ARTIFACTS_PATH", "/unused")
    assert harness.main(["--artifacts", str(tmp_path / "missing"), "--output", str(output)]) == 1
    result = json.loads((output / "result.json").read_text())
    assert result["phase"] == "verify"
    assert "skip numerical" in result["error"]
    assert "failure" in (output / "junit.xml").read_text()


def test_setup_copies_local_and_preserves_source(tmp_path, monkeypatch):
    root = artifacts(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls = []

    class OriginalTest:
        RBLN_CLASS = SimpleNamespace(from_pretrained=lambda path, **kwargs: calls.append((path, kwargs)))
        get_rbln_local_dir = classmethod(lambda cls: "local-test-artifact")
        setUpClass = harness.frozen_setup(root / "encoder")

    OriginalTest.setUpClass()
    destination = Path("local-test-artifact").resolve()
    assert calls == [(destination, {"rbln_device": 0})]
    assert (destination / "embedding.rbln").read_bytes() == (root / "encoder" / "embedding.rbln").read_bytes()
    assert destination != root / "encoder"


def test_stage_records_cold_warm_and_errors_without_changing_returns(tmp_path):
    recorder = harness.Recorder(tmp_path)
    seen = []
    result = object()
    stage = recorder.stage(lambda value: seen.append(value) or result, "encoder_0", 2)
    assert stage("one") is result
    assert stage("two") is result
    assert seen == ["one", "two"]

    def fail():
        raise RuntimeError("device timeout")

    with pytest.raises(RuntimeError, match="device timeout"):
        recorder.stage(fail, "tail", 2)()
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in rows] == ["stage_start", "stage_end"] * 3
    assert [row["temperature"] for row in rows[1::2]] == ["cold", "warm", "cold"]
    assert "device timeout" in rows[-1]["error"]
    assert rows[-1]["seconds"] >= 0


def test_harness_preserves_original_methods_and_instruments_reload(tmp_path, monkeypatch):
    from optimum.rbln.transformers.models.dinov2.modeling_dinov2 import Dinov2Mixin
    from tests import test_transformers

    methods = {
        name: {
            method: getattr(getattr(test_transformers, name), method)
            for method in dir(getattr(test_transformers, name))
            if method.startswith("test_")
        }
        for name in harness.CLASSES
    }
    recorder = harness.Recorder(tmp_path)
    marker = object()
    configs = SimpleNamespace(compile_cfgs=[SimpleNamespace(compiled_model_name=name) for name in harness.STAGES])
    chains = []

    def original(cls, compiled_models, config):
        chain = SimpleNamespace(runtimes=[lambda *_: marker for _ in harness.STAGES], lock=marker)
        chains.append(chain)
        return [chain]

    monkeypatch.setattr(Dinov2Mixin, "_create_runtimes", classmethod(original))

    def pytest_main(args, plugins):
        assert [arg for arg in args if "::" in arg] == [
            f"tests/test_transformers.py::{name}" for name in harness.CLASSES
        ]
        for name in harness.CLASSES:
            cls = getattr(test_transformers, name)
            assert len(methods[name]) == 9
            for method, implementation in methods[name].items():
                assert getattr(cls, method) is implementation
            for _ in range(2):
                chain = cls.RBLN_CLASS._create_runtimes([], configs)[0]
                assert chain.lock is marker
                assert chain.runtimes[0]("input") is marker
        assert plugins == [recorder]
        return 0

    monkeypatch.setattr(pytest, "main", pytest_main)
    assert harness.run_tests(tmp_path, recorder) == 0
    assert recorder.loads == 4
    assert len(chains) == 4
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len([row for row in rows if row["phase"] == "stage_end"]) == 4
