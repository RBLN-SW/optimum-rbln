"""CPU controls for the frozen GELU diagnostic; no compiler or device is used."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.diagnostics import siglip2_gelu_frozen as diagnostic


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    manifest = {
        "source_compiler": diagnostic.COMPILER,
        "source_optimum": diagnostic.OPTIMUM,
        "frontend_sha256": diagnostic.FRONTEND_SHA,
        "mapped_librbln_sha256": diagnostic.LIBRARY_SHA,
        "files": {},
    }
    binaries = {}
    metadata = {}
    for variant in ("before", "after"):
        directory = tmp_path / variant
        directory.mkdir()
        config = {
            "batch_size": 1,
            "text_batch_size": 1,
            "max_seq_len": 64,
            "max_patches": 256,
            "dtype": "float32",
            "_compile_cfgs": [
                {
                    "compiled_model_name": "compiled_model",
                    "input_info": diagnostic.INPUTS,
                    "npu": "RBLN-CA25",
                    "num_devices": 1,
                }
            ],
        }
        for name in diagnostic.FILES:
            data = {"model_type": "siglip2", "vision_config": {"num_patches": 256}}
            (directory / name).write_text(json.dumps(config if name == "rbln_config.json" else data))
            path = directory / name
            manifest["files"][f"{variant}/{name}"] = {"sha256": diagnostic.sha256(path), "bytes": path.stat().st_size}
        receipt = manifest["files"][f"{variant}/compiled_model.rbln"]
        binaries[variant] = (receipt["sha256"], receipt["bytes"])
        metadata[variant] = {
            "npu": "RBLN-CA25",
            "num_devices": 1,
            "inputs": [{"name": name, "shape": shape, "dtype": dtype} for name, shape, dtype in diagnostic.INPUTS],
            "outputs": [{"shape": shape, "dtype": "float32"} for shape in diagnostic.OUTPUT_SHAPES],
            "subgraph": {"stats": [{"device_func_count": 3}]},
        }
    monkeypatch.setattr(diagnostic, "BINARIES", binaries)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, manifest, metadata


def test_verification_accepts_exact_frozen_interfaces_without_runtime(frozen):
    root, _, metadata = frozen
    receipt = diagnostic.verify_artifacts(root, lambda path: metadata[path.parent.name])
    assert set(receipt["metadata"]) == {"before", "after"}


@pytest.mark.parametrize("damage", ["checksum", "source", "extra_file", "input_shape", "partition_count", "symlink"])
def test_verification_rejects_wrong_frozen_evidence(frozen, damage):
    root, manifest, metadata = frozen
    if damage == "checksum":
        (root / "before" / "tokenizer.json").write_text("changed")
    elif damage == "source":
        manifest["source_compiler"] = "wrong"
        (root / "manifest.json").write_text(json.dumps(manifest))
    elif damage == "extra_file":
        (root / "after" / "extra.json").write_text("{}")
    elif damage == "input_shape":
        metadata["after"]["inputs"][0]["shape"] = [8, 64]
    elif damage == "partition_count":
        metadata["after"]["subgraph"]["stats"][0]["device_func_count"] = 1
    else:
        path = root / "before" / "tokenizer.json"
        path.unlink()
        path.symlink_to(root / "after" / "tokenizer.json")
    with pytest.raises(ValueError, match=r"checksum|source identity|twelve|binary inputs|partition count|Unsafe"):
        diagnostic.verify_artifacts(root, lambda path: metadata[path.parent.name])


def cases():
    result = []
    for case in range(4):
        expected = tuple(
            torch.tensor([[float(case + 1)]]) if i < 2 else torch.arange(1, 9).float().reshape(1, 8) + case
            for i in range(8)
        )
        result.append(({"case": torch.tensor([case])}, expected))
    return result


def fake_loader(inputs, bad_after=False, unstable_reload=False):
    loads = []

    def loader(directory):
        variant = directory.name
        load_index = loads.count(variant)
        loads.append(variant)

        def model(**kwargs):
            index = int(kwargs["case"].item())
            values = list(copy.deepcopy(inputs[index][1]))
            if variant == "before" or (bad_after and index == 3):
                values[6] *= 0
            if unstable_reload and variant == "after" and load_index == 1:
                values[7] += 0.01
            return SimpleNamespace(to_tuple=lambda: (*values[:4], tuple(values[4:6]), tuple(values[6:])))

        return model

    return loader, loads


@pytest.mark.parametrize("bad_after", [False, True])
def test_all_cases_and_baseline_failures_are_saved_before_after_gate(tmp_path, bad_after):
    inputs = cases()
    loader, loads = fake_loader(inputs, bad_after=bad_after)
    result = diagnostic.run_comparison(tmp_path / "artifacts", tmp_path, inputs, loader)
    assert loads == ["before", "before", "after", "after"]
    assert result["before_failed_criteria"]
    assert result["passed"] is not bad_after
    assert bool(result["after_failed_criteria"]) is bad_after
    for variant in ("before", "after"):
        for phase in ("initial", "reload"):
            for case in range(4):
                saved = torch.load(tmp_path / variant / f"{phase}-case{case}.pt", weights_only=True)
                assert set(saved) == {"inputs", "actual", "expected", "repeated"}
                assert len(saved["actual"]) == len(saved["expected"]) == len(saved["repeated"]) == 8
            assert (tmp_path / variant / f"{phase}-return-to-first.pt").is_file()
    timings = json.loads((tmp_path / "timings.json").read_text())
    assert len([event for event in timings if "host_wall_seconds" in event]) == 36
    assert (tmp_path / "results.json").is_file()


def test_reload_exactness_is_required_even_when_parity_passes(tmp_path):
    inputs = cases()
    loader, _ = fake_loader(inputs, unstable_reload=True)
    result = diagnostic.run_comparison(tmp_path / "artifacts", tmp_path, inputs, loader)
    assert result["passed"] is False
    assert any("changed outputs" in message for message in result["after_failed_criteria"])


def test_outputs_are_saved_before_metrics_are_called(tmp_path, monkeypatch):
    original = diagnostic.output_metrics
    inputs = cases()
    loader, _ = fake_loader(inputs)
    calls = 0

    def checked_metrics(actual, expected, repeated):
        nonlocal calls
        variant = "before" if calls < 8 else "after"
        phase = "initial" if calls % 8 < 4 else "reload"
        index = calls % 4
        assert (tmp_path / variant / f"{phase}-case{index}.pt").is_file()
        calls += 1
        return original(actual, expected, repeated)

    monkeypatch.setattr(diagnostic, "output_metrics", checked_metrics)
    diagnostic.run_comparison(tmp_path / "artifacts", tmp_path, inputs, loader)
    assert calls == 16


def test_scalar_logits_skip_pcc_but_hidden_and_embeddings_require_it():
    expected = cases()[0][1]
    rows = diagnostic.output_metrics(expected, expected, expected)
    assert all(not row["failed_criteria"] for row in rows)
    assert all("pcc" not in row for row in rows[:2])
    assert all("pcc" in row for row in rows[2:])
    assert rows[2]["minimum_cosine"] >= 0.99
    actual = list(copy.deepcopy(expected))
    actual[6][0, 0] = torch.nan
    rows = diagnostic.output_metrics(actual, expected, actual)
    assert "finite" in rows[6]["failed_criteria"]


def test_verify_only_cli_never_builds_inputs_or_creates_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["siglip2_gelu_frozen", "--artifacts", str(tmp_path), "--out", str(tmp_path / "out"), "--verify-only"],
    )
    monkeypatch.setattr(diagnostic, "verify_artifacts", lambda root: {"manifest": {}, "metadata": {}})
    monkeypatch.setattr(diagnostic.importlib.metadata, "version", lambda name: diagnostic.SDK)
    monkeypatch.setattr(
        diagnostic.subprocess, "check_output", lambda *args, **kwargs: "validation-head" if kwargs.get("text") else b""
    )

    def forbidden(*args, **kwargs):
        pytest.fail("verify-only must not load CPU oracle or device models")

    monkeypatch.setattr(diagnostic, "native_cases", forbidden)
    monkeypatch.setattr(diagnostic, "run_comparison", forbidden)
    diagnostic.main()
    assert (tmp_path / "out" / "verification.json").is_file()


def test_baseline_repeat_failure_also_fails_final_gate(tmp_path):
    inputs = cases()
    original, _ = fake_loader(inputs)

    def loader(directory):
        model = original(directory)
        calls = 0

        def unstable(**kwargs):
            nonlocal calls
            output = model(**kwargs)
            values = list(diagnostic.flatten(output))
            if directory.name == "before" and calls % 2 == 1:
                values[0] += 0.001
            calls += 1
            return SimpleNamespace(to_tuple=lambda: (*values[:4], tuple(values[4:6]), tuple(values[6:])))

        return unstable

    result = diagnostic.run_comparison(tmp_path / "artifacts", tmp_path, inputs, loader)
    assert result["passed"] is False
    assert result["before_stability_failures"]
    assert not result["after_failed_criteria"]
