import ast
import hashlib
import inspect
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.diagnostics import mpnet_b2048_frozen as harness


def artifacts(tmp_path, monkeypatch):
    root = tmp_path / "frozen"
    root.mkdir()
    model = {"model_type": "mpnet", "num_hidden_layers": 12, "hidden_size": 768, "max_position_embeddings": 514}
    inputs = [["input_ids", [2048, 512], "int64"], ["attention_mask", [2048, 512], "int64"]]
    config = {
        "batch_size": 2048,
        "max_seq_len": 512,
        "_compile_cfgs": [
            {
                "compiled_model_name": "compiled_model",
                "input_info": inputs,
                "npu": "RBLN-CR13",
                "num_devices": 1,
            }
        ],
    }
    (root / "config.json").write_text(json.dumps(model))
    (root / "rbln_config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "tokenizer_config.json", "compiled_model.rbln"):
        (root / name).write_text(name)
    manifest = {
        "runtime_sdk": harness.SDK,
        "compiler_kind": "source-hybrid root4b + AA + f66 HComm",
        "compiler_dso_sha256": harness.HYBRID_DSO,
        "checkpoint": harness.CHECKPOINT,
        "revision": harness.REVISION,
        "npu": "RBLN-CR13",
        "batch_size": 2048,
        "max_seq_len": 512,
        "reviewed_product_head": harness.PRODUCT,
        **harness.SOURCE_CONTRACT,
        "device_alloc_bytes": 59856912384,
        "files": {},
    }
    for path in root.iterdir():
        manifest["files"][path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    monkeypatch.setattr(harness, "COMPILED_SHA256", manifest["files"]["compiled_model.rbln"]["sha256"])
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def metadata(path):
    return {
        "npu": "RBLN-CR13",
        "num_devices": 1,
        "uuid": "session",
        "compiler_version": harness.COMPILER_LABEL,
        "alloc_per_node": [59856912384],
        "inputs": [{"name": name, "shape": [2048, 512], "dtype": "int64"} for name in ("input_ids", "attention_mask")],
        "outputs": [{"shape": shape, "dtype": "float32"} for shape in ([2048, 512, 768], [2048, 768])],
    }


def test_hashes_and_native_metadata_and_honest_provenance(tmp_path, monkeypatch):
    root = artifacts(tmp_path, monkeypatch)
    result = harness.verify_artifacts(root, inspect_model=metadata)
    assert result["compile_checkout_head_captured"] is True
    assert result["manifest"]["reviewed_product_head"] == harness.PRODUCT
    bad = metadata(None)
    bad["inputs"][0]["shape"] = [1024, 512]
    with pytest.raises(ValueError, match="metadata"):
        harness.verify_artifacts(root, inspect_model=lambda path: bad)
    (root / "compiled_model.rbln").write_text("corrupt")
    with pytest.raises(ValueError, match="Checksum"):
        harness.verify_artifacts(root, inspect_model=metadata)


def test_verify_only_never_loads_or_tests(tmp_path, monkeypatch):
    root = artifacts(tmp_path, monkeypatch)
    original = harness.verify_artifacts
    monkeypatch.setattr(harness, "verify_artifacts", lambda directory: original(directory, inspect_model=metadata))
    monkeypatch.setattr(harness.importlib.metadata, "version", lambda name: harness.SDK)
    monkeypatch.setattr(harness, "run_tests", lambda *args: pytest.fail("runtime execution"))
    output = tmp_path / "output"
    assert harness.main(["--artifacts", str(root), "--output", str(output), "--verify-only"]) == 0
    result = json.loads((output / "result.json").read_text())
    assert result["accepted"] is False
    assert result["runtime_executed"] is False
    assert result["phase"] == "verified_only"
    assert (output / "junit.xml").is_file()


def test_skip_environment_is_rejected_with_failure_receipts(tmp_path, monkeypatch):
    monkeypatch.setenv("SAVE_ARTIFACTS_PATH", "unused")
    output = tmp_path / "output"
    assert harness.main(["--artifacts", str(tmp_path), "--output", str(output)]) == 1
    assert "skips original numerical" in json.loads((output / "result.json").read_text())["error"]
    assert "failure" in (output / "junit.xml").read_text()


def test_original_methods_and_all_four_texts_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("RBLN_TEST_NPU", "RBLN-CR13")
    from tests import test_transformers

    parent = test_transformers.TestMPNetModelNativeMaxBatch1024
    recorder = harness.Recorder(tmp_path)
    case = harness.make_case(parent, tmp_path, recorder)
    assert len(recorder.expected_methods) == 10
    for method in recorder.expected_methods - {"test_repeat_reload_tuple"}:
        assert getattr(case, method) is getattr(parent, method)
    function = ast.parse(textwrap.dedent(inspect.getsource(parent.test_embedding_parity)))
    texts = next(
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "texts" for target in node.targets)
    )
    original_texts = [
        item.left.value * item.right.value if isinstance(item, ast.BinOp) else item.value for item in texts.elts
    ]
    assert original_texts == harness.TEXTS
    assert case.BATCH_SIZE == 2048
    assert case.GENERATION_KWARGS["input_ids"].shape == (2048, 512)
    assert parent.BATCH_SIZE == 1024
    assert case.HF_CONFIG_KWARGS == parent.HF_CONFIG_KWARGS


def test_collection_guard_rejects_missing_original_methods(tmp_path):
    recorder = harness.Recorder(tmp_path)
    recorder.expected_methods = {"test_embedding_parity", "test_generate"}
    item = SimpleNamespace(cls=type(harness.CASE, (), {}), originalname="test_generate")
    with pytest.raises(ValueError, match="every inherited"):
        recorder.pytest_collection_modifyitems([item])


def test_local_copy_setup_cannot_recompile_or_delete_frozen_source(tmp_path, monkeypatch):
    root = artifacts(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    calls = []

    class Case:
        get_rbln_local_dir = classmethod(lambda cls: "local-artifact")
        RBLN_CLASS = SimpleNamespace(from_pretrained=lambda path, **kwargs: calls.append((path, kwargs)))
        setUpClass = harness.frozen_setup(root, harness.Recorder(tmp_path))

    Case.setUpClass()
    assert calls == [(Path("local-artifact").resolve(), {"rbln_device": 0})]
    assert (root / "compiled_model.rbln").is_file()


@pytest.mark.parametrize("corrupt_output", [None, 0, 1])
def test_strict_repeat_reload_tuple_covers_four_texts_and_both_outputs(tmp_path, monkeypatch, corrupt_output):
    import transformers

    batches = []
    loads = []

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            batches.append((texts, kwargs))
            return {"input_ids": torch.zeros(2048, 512, dtype=torch.int64), "attention_mask": torch.ones(2048, 512)}

    class Output:
        def __init__(self, corrupt):
            self.values = [torch.ones(2, 3), torch.ones(2, 1)]
            if corrupt is not None:
                self.values[corrupt].add_(1)

        def to_tuple(self):
            return tuple(self.values)

    class Model:
        def __init__(self, corrupt=None):
            self.corrupt = corrupt

        def __call__(self, **kwargs):
            output = Output(self.corrupt)
            return output if kwargs.get("return_dict") else output.to_tuple()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())

    class Case:
        BATCH_SIZE = 2048
        DEVICE = 0
        HF_MODEL_ID = harness.CHECKPOINT
        HF_CONFIG_KWARGS = {"revision": harness.REVISION}
        model = Model()
        RBLN_AUTO_CLASS = SimpleNamespace(
            from_pretrained=lambda path, **kwargs: loads.append((path, kwargs)) or Model(corrupt_output)
        )
        get_rbln_local_dir = classmethod(lambda cls: "local-artifact")

    if corrupt_output is None:
        harness.repeat_reload_tuple(Case())
    else:
        with pytest.raises(AssertionError):
            harness.repeat_reload_tuple(Case())
    assert set(batches[0][0]) == set(harness.TEXTS)
    assert len(batches[0][0]) == 2048
    assert batches[0][1]["max_length"] == 512
    assert loads == [("local-artifact", {"rbln_device": 0})]


@pytest.mark.parametrize("outcome", ["skip", "failure", "missing_method"])
def test_partial_test_run_never_reports_acceptance(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(harness.importlib.metadata, "version", lambda name: harness.SDK)
    monkeypatch.setattr(harness, "verify_artifacts", lambda path: {})
    monkeypatch.setattr(harness, "disk_receipts", dict)

    def partial(source, recorder):
        recorder.collected = 9 if outcome == "missing_method" else 10
        recorder.skipped = ["test_embedding_parity"] if outcome == "skip" else []
        recorder.failed = ["test_generate"] if outcome == "failure" else []
        return 0

    monkeypatch.setattr(harness, "run_tests", partial)
    output = tmp_path / "output"
    assert harness.main(["--artifacts", str(tmp_path), "--output", str(output)]) == 1
    result = json.loads((output / "result.json").read_text())
    assert result["accepted"] is False
    assert result["runtime_executed"] is False


def test_numerical_success_cannot_accept_known_learned_cpu_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(harness.importlib.metadata, "version", lambda name: harness.SDK)
    monkeypatch.setattr(harness, "verify_artifacts", lambda path: {})
    monkeypatch.setattr(harness, "disk_receipts", dict)

    def complete(source, recorder):
        recorder.collected = 10
        recorder.runtime_calls = 4
        return 0

    monkeypatch.setattr(harness, "run_tests", complete)
    output = tmp_path / "output"
    assert harness.main(["--artifacts", str(tmp_path), "--output", str(output)]) == 0
    result = json.loads((output / "result.json").read_text())
    assert result["numerical_tests_passed"] is True
    assert result["runtime_executed"] is True
    assert result["accepted"] is False
    assert result["full_device_acceptance"] is False
    assert "learned word and position embedding" in result["full_device_acceptance_blocker"]


@pytest.mark.parametrize("field", [*harness.SOURCE_CONTRACT, "compiler_dso_sha256", "runtime_sdk"])
def test_captured_hybrid_provenance_is_required(tmp_path, monkeypatch, field):
    root = artifacts(tmp_path, monkeypatch)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest[field] = "wrong source/runtime receipt"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="interface/source"):
        harness.verify_artifacts(root, inspect_model=metadata)


def test_runtime_version_is_separate_from_captured_compiler_label(tmp_path, monkeypatch):
    root = artifacts(tmp_path, monkeypatch)
    assert harness.SDK != harness.COMPILER_LABEL
    wrong = metadata(None)
    wrong["compiler_version"] = harness.SDK
    with pytest.raises(ValueError, match="hybrid compiler label"):
        harness.verify_artifacts(root, inspect_model=lambda path: wrong)
