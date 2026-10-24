import hashlib
import json

import pytest
import torch

from scripts.diagnostics import modernbert_first_layer as diagnostic

from .test_modernbert import tiny_model


@pytest.mark.parametrize(("batch", "length"), [(2, 17), (1, 8192)])
def test_all_six_outputs_match_original_forward_and_actual_residual_input(batch, length):
    native = tiny_model()
    ids = torch.randint(1, 128, (batch, length))
    mask = torch.ones_like(ids)
    mask[:, -3:] = 0
    with torch.inference_mode():
        expected = diagnostic.original_outputs(native, {"input_ids": ids, "attention_mask": mask})
        actual = diagnostic.FirstLayerWrapper(native)(ids, mask)
    assert len(actual) == len(diagnostic.OUTPUT_NAMES) == 6
    for value, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, target, rtol=0, atol=0)
    # These are the real native residual operands, not a similarly named norm/mask addition.
    torch.testing.assert_close(actual[3], actual[1] + actual[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[5], actual[3] + actual[4], rtol=0, atol=0)
    assert not native.embeddings._forward_hooks
    assert not native.layers[0].mlp_norm._forward_pre_hooks


def test_first_layer_identity_rejects_a_changed_tensor_even_with_same_shape(monkeypatch):
    frozen = torch.zeros(1, 8192, 768)
    monkeypatch.setattr(diagnostic, "LAYER0_SHA256", hashlib.sha256(frozen.numpy().tobytes()).hexdigest())
    assert diagnostic.compare_first_layer_to_frozen(frozen)["bit_exact"]
    changed = frozen.clone()
    changed[0, 123, 7] = 1e-6
    assert not diagnostic.compare_first_layer_to_frozen(changed)["bit_exact"]


@pytest.mark.parametrize("value", [torch.zeros(1, 17, 8), torch.zeros(1, 8192, 768, dtype=torch.float16)])
def test_first_layer_identity_requires_native_geometry_and_original_dtype(value):
    with pytest.raises(ValueError, match="native F32"):
        diagnostic.compare_first_layer_to_frozen(value)


@pytest.mark.parametrize("variable", ["SAVE_ARTIFACTS_PATH", "REUSE_ARTIFACTS_PATH"])
def test_artifact_compatibility_is_rejected_before_loading(tmp_path, monkeypatch, variable):
    monkeypatch.setenv(variable, "compat")
    monkeypatch.setattr(diagnostic, "load_baseline", lambda *args: pytest.fail("Baseline loaded"))
    assert diagnostic.main(["--compile-only", "--baseline", str(tmp_path / "missing"), "--output", str(tmp_path)]) == 1
    assert "compatibility mode" in json.loads((tmp_path / "result.json").read_text())["error"]


def test_execute_requires_scheduler_before_loading(tmp_path, monkeypatch):
    monkeypatch.delenv("BUILDKITE_BUILD_ID", raising=False)
    monkeypatch.setattr(diagnostic, "load_baseline", lambda *args: pytest.fail("Baseline loaded"))
    assert diagnostic.main(["--execute", "--baseline", str(tmp_path / "missing"), "--output", str(tmp_path)]) == 1
    assert "BUILDKITE_BUILD_ID" in json.loads((tmp_path / "result.json").read_text())["error"]


def test_changed_cpu_reference_stops_before_compilation(tmp_path, monkeypatch):
    import transformers

    from optimum.rbln import RBLNModernBertModel

    native = tiny_model()
    inputs = {
        "input_ids": torch.ones(1, 17, dtype=torch.int64),
        "attention_mask": torch.ones(1, 17, dtype=torch.int64),
    }
    with torch.inference_mode():
        frozen = native(**inputs).last_hidden_state.clone()
    # A tiny reference change must reject causal use even though numerical tolerance would pass.
    frozen = frozen.clone()
    frozen[0, 3, 2] += 1e-6
    parameter_count = torch.zeros(1)
    monkeypatch.setattr(parameter_count, "numel", lambda: 149014272)
    monkeypatch.setattr(native, "parameters", lambda: iter([parameter_count]))
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: (native, {"missing_keys": [], "unexpected_keys": []}),
    )
    monkeypatch.setattr(diagnostic, "verify_trained_state", lambda model: {"exact_checkpoint_tensors": True})
    monkeypatch.setattr(diagnostic, "load_baseline", lambda path: {"inputs": inputs, "reference": frozen})
    monkeypatch.setattr(diagnostic.importlib.metadata, "version", lambda name: diagnostic.SDK)
    monkeypatch.setattr(
        RBLNModernBertModel, "from_pretrained", lambda *args, **kwargs: pytest.fail("Compiler reached")
    )
    monkeypatch.delenv("SAVE_ARTIFACTS_PATH", raising=False)
    monkeypatch.delenv("REUSE_ARTIFACTS_PATH", raising=False)
    assert diagnostic.main(["--compile-only", "--baseline", str(tmp_path / "frozen"), "--output", str(tmp_path)]) == 1
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["phase"] == "cpu_reference"
    assert not result["cpu_reference_against_frozen"]["bit_exact"]
    assert "CPU reference changed" in result["error"]
