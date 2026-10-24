"""Frozen diagnostic guards and reload-object ownership, without device access."""

import json
from types import SimpleNamespace

import pytest
import torch

from scripts.diagnostics.chronos2_frozen_validation import (
    NAMES,
    PRIMITIVE,
    SOURCE,
    STAGED_SHA,
    block_export,
    runtime_adapter,
    sha256,
    validate_frozen,
)


@pytest.fixture
def bundle(tmp_path):
    hidden, mask, groups = [512, 577, 768], [512, 1, 1, 577], [577, 1, 512, 512]
    native_inputs = [
        ["patched_context", [512, 512, 48], "float32"],
        ["context_mask", [512, 512], "bool"],
        ["patched_future", [512, 64, 48], "float32"],
        ["group_ids", [512], "int64"],
    ]
    (tmp_path / "original_rbln_config.json").write_text(
        json.dumps(
            {
                "cls_name": "RBLNChronos2ModelConfig",
                "batch_size": 512,
                "context_length": 8192,
                "prediction_length": 1024,
                "_compile_cfgs": [{"input_info": native_inputs}],
            }
        )
    )
    (tmp_path / "original-test-generate-inputs.pt").write_bytes(b"frozen fixture")
    stages, configs = [], []
    for name in NAMES:
        inputs = native_inputs if name == "embedding" else [["hidden", hidden, "float32"]]
        if name.startswith("encoder_"):
            inputs += [["attention_mask", mask, "float32"], ["group_time_mask", groups, "float32"]]
        outputs = [hidden, mask, groups] if name == "embedding" else [hidden]
        if name == "forecast":
            outputs = [[512, 21, 1024], [512, 21, 1024]]
        path = tmp_path / f"{name}.rbln"
        path.write_bytes(name.encode())
        stages.append(
            {
                "name": name,
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "status": "compiled",
                "alloc_per_node": [123],
                "metadata": {
                    "npu": "RBLN-CA25",
                    "num_devices": 1,
                    "inputs": [{"name": n, "shape": s, "dtype": d} for n, s, d in inputs],
                    "outputs": [{"shape": s, "dtype": "float32"} for s in outputs],
                    "subgraph": {"stats": [{"compile_id": 0, "device_func_count": 1, "host_op_reason": {}}]},
                },
            }
        )
        configs.append({"compiled_model_name": name, "input_info": inputs, "npu": "RBLN-CA25", "num_devices": 1})
    data = {
        "status": "compiled",
        "compile_only": True,
        "runtime_executed": False,
        "source_compiler": SOURCE,
        "source_optimum": "bbc20bad08c895d6179128cb9efe8862c85726ba",
        "frontend_sha256": "522ccfdcaf665165ad9f2e8a4a45a53a9d92d2b4c6a766529aa4545db78e8e1e",
        "mapped_librbln_sha256": STAGED_SHA,
        "primitive": PRIMITIVE,
        "checkpoint_revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
        "batch_size": 512,
        "context_length": 8192,
        "prediction_length": 1024,
        "encoder_chunk_size": 3,
        "frozen_inputs_exact_reused": True,
        "staged_logging_provenance": {
            "source_base": "5f544486937380881f229e48ed9b2161f243e265",
            "logging_diff_sha256": "8ad4826b3ddef7408374354c005f5d415077b951cf913876bcd61f6272e6af70",
        },
        "activation": {"gelu_module_count": 0},
        "original_input_file_sha256": sha256(tmp_path / "original-test-generate-inputs.pt"),
        "stages": stages,
        "compile_configs": configs,
    }
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "compile_configs.json").write_text(json.dumps(configs))
    data["files"] = {
        name: {"bytes": (tmp_path / name).stat().st_size, "sha256": sha256(tmp_path / name)}
        for name in ("config.json", "original_rbln_config.json", "compile_configs.json")
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return tmp_path, data


def test_artifact_hash_rejected_before_inspector_or_runtime(bundle):
    root, _ = bundle
    (root / "encoder_0.rbln").write_bytes(b"altered")
    with pytest.raises(ValueError, match="SHA256/size"):
        validate_frozen(root, sha256(root / "manifest.json"))


def test_joint_group_geometry_cannot_be_reduced_even_with_consistent_config(bundle):
    root, data = bundle
    for container in (
        data["stages"][1]["metadata"]["inputs"][2]["shape"],
        data["compile_configs"][1]["input_info"][2][1],
    ):
        container[-1] = 256
    (root / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Full joint native geometry"):
        validate_frozen(root, sha256(root / "manifest.json"))


def test_actual_inspected_interface_must_match_frozen_manifest(bundle):
    root, data = bundle
    validate_frozen(root, sha256(root / "manifest.json"))

    def altered(path):
        return {**data["stages"][NAMES.index(path.stem)]["metadata"], "num_devices": 2}

    with pytest.raises(ValueError, match="Actual compiled interface"):
        validate_frozen(root, sha256(root / "manifest.json"), altered)


def test_export_is_blocked_before_remote_model_or_compiler_access(monkeypatch):
    from optimum.rbln import RBLNChronos2Model

    monkeypatch.setattr(RBLNChronos2Model, "_export", classmethod(block_export))
    with pytest.raises(RuntimeError, match="forbids model export"):
        RBLNChronos2Model.from_pretrained("unused-remote-model", export=True)


def test_fresh_reload_uses_passed_objects_and_releases_runtime_before_next_stage(bundle):
    _, manifest = bundle
    active, calls, events = [], [], []

    class Runtime:
        def __init__(self, owner):
            assert not active
            self.owner = owner
            active.append(owner)

        def __call__(self, *inputs):
            if self.owner[1] == "embedding":
                return inputs[0] + 1, torch.zeros(1), torch.zeros(1)
            if self.owner[1] == "forecast":
                return inputs[0] + 1, inputs[0] + 2
            return inputs[0] + 1

        def __del__(self):
            active.remove(self.owner)

    class Compiled:
        def __init__(self, generation, name):
            self.owner = generation, name

        def create_runtime(self, **kwargs):
            calls.append(self.owner)
            assert kwargs["device"] == 0
            assert kwargs["tensor_type"] == "pt"
            return Runtime(self.owner)

    config = SimpleNamespace(
        compile_cfgs=[SimpleNamespace(compiled_model_name=n) for n in NAMES],
        device_map=dict.fromkeys(NAMES, 0),
        activate_profiler=False,
        timeout=60000,
    )
    original = [Compiled("original", name) for name in NAMES]
    reloaded = [Compiled("reloaded", name) for name in NAMES]
    adapter = runtime_adapter(manifest, events.append)
    first = adapter(None, original, config)[0]
    fresh = adapter(None, reloaded, config)[0]
    for chain in (first, fresh):
        output = chain(torch.zeros(1))
        torch.testing.assert_close(output[0], torch.tensor([6.0]))
        torch.testing.assert_close(output[1], torch.tensor([7.0]))
        assert not active
    assert calls == [(generation, name) for generation in ("original", "reloaded") for name in NAMES]


def test_stage_bundle_roundtrips_actual_config_api_with_six_names(bundle, tmp_path):
    from optimum.rbln import RBLNChronos2ModelConfig
    from scripts.diagnostics.chronos2_frozen_validation import stage_bundle

    root, manifest = bundle
    (root / "config.json").write_text("{}")
    destination = tmp_path / "loaded-six-stage-bundle"
    stage_bundle(root, destination, manifest)
    actual = RBLNChronos2ModelConfig.from_pretrained(destination)
    original = RBLNChronos2ModelConfig.from_pretrained(root / "original_rbln_config.json")
    assert [cfg.compiled_model_name for cfg in actual.compile_cfgs] == list(NAMES)
    assert json.loads(json.dumps([cfg.input_info for cfg in actual.compile_cfgs])) == [
        cfg["input_info"] for cfg in manifest["compile_configs"]
    ]
    serialized = json.loads((destination / "rbln_config.json").read_text())
    assert serialized["_compile_cfgs"] == manifest["compile_configs"]
    # Loading normalizes runtime options identically for both original and cloned configs.
    assert all(cfg.npu == original.compile_cfgs[0].npu for cfg in actual.compile_cfgs)
    assert actual.batch_size == original.batch_size == 512
    assert actual.context_length == original.context_length == 8192
    assert actual.prediction_length == original.prediction_length == 1024
    assert actual.dtype == original.dtype
