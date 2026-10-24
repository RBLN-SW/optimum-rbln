import pytest
import torch
from transformers import ModernBertModel

from scripts.diagnostics.modernbert_boundaries import BoundaryWrapper, compare_to_frozen

from .test_modernbert import tiny_model


@pytest.mark.parametrize("length", [17, 8192])
def test_boundaries_are_exact_original_layer_outputs_and_final(length):
    config = tiny_model().config
    config.num_hidden_layers = 3
    config.layer_types = ["full_attention", "sliding_attention", "sliding_attention"]
    native = ModernBertModel(config).eval()
    recorded = {}
    for index, layer in enumerate(native.layers):
        layer.register_forward_hook(lambda module, args, output, index=index: recorded.update({index: output.clone()}))
    ids = torch.randint(1, 128, (1, length))
    mask = torch.ones_like(ids)
    mask[:, -3:] = 0
    with torch.inference_mode():
        final = native(input_ids=ids, attention_mask=mask).last_hidden_state
        expected = {i: value.clone() for i, value in recorded.items()}
        actual = BoundaryWrapper(native, (0, 1))(ids, mask)
    assert len(actual) == 3
    torch.testing.assert_close(actual[0], final, rtol=0, atol=0)
    for index in (0, 1):
        torch.testing.assert_close(actual[index + 1], expected[index], rtol=0, atol=0)


def test_changed_final_invalidates_instrumentation_identity():
    frozen = torch.randn(1, 5, 8)
    assert compare_to_frozen(frozen.clone(), frozen)["bit_exact"]
    changed = frozen.clone()
    changed[0, 1, 2] += 1e-6
    assert not compare_to_frozen(changed, frozen)["bit_exact"]


def test_rejects_hf_final_state_that_is_already_normalized():
    with pytest.raises(ValueError, match="HF replaces"):
        BoundaryWrapper(tiny_model(), (1,))


@pytest.mark.parametrize("variable", ["SAVE_ARTIFACTS_PATH", "REUSE_ARTIFACTS_PATH"])
def test_artifact_compatibility_is_rejected_before_loading(tmp_path, monkeypatch, variable):
    import json

    from scripts.diagnostics import modernbert_boundaries as diagnostic

    monkeypatch.setenv(variable, "compat")
    monkeypatch.setattr(diagnostic, "load_baseline", lambda *args: pytest.fail("Baseline loaded"))
    assert diagnostic.main(["--compile-only", "--baseline", str(tmp_path / "missing"), "--output", str(tmp_path)]) == 1
    assert "compatibility mode" in json.loads((tmp_path / "result.json").read_text())["error"]


def test_execute_requires_scheduler_before_loading(tmp_path, monkeypatch):
    import json

    from scripts.diagnostics import modernbert_boundaries as diagnostic

    monkeypatch.delenv("BUILDKITE_BUILD_ID", raising=False)
    monkeypatch.setattr(diagnostic, "load_baseline", lambda *args: pytest.fail("Baseline loaded"))
    assert diagnostic.main(["--execute", "--baseline", str(tmp_path / "missing"), "--output", str(tmp_path)]) == 1
    assert "BUILDKITE_BUILD_ID" in json.loads((tmp_path / "result.json").read_text())["error"]


def test_early_boundaries_keep_final_and_selected_original_layers():
    from scripts.diagnostics.modernbert_boundaries import BOUNDARIES

    assert BOUNDARIES == (0, 4, 9, 14)
    config = tiny_model().config
    config.num_hidden_layers = 16
    config.layer_types = ["full_attention" if index % 3 == 0 else "sliding_attention" for index in range(16)]
    native = ModernBertModel(config).eval()
    recorded = {}
    for index in BOUNDARIES:
        native.layers[index].register_forward_hook(
            lambda module, args, output, index=index: recorded.update({index: output.clone()})
        )
    ids = torch.randint(1, 128, (1, 17))
    mask = torch.ones_like(ids)
    mask[:, -3:] = 0
    with torch.inference_mode():
        final = native(input_ids=ids, attention_mask=mask).last_hidden_state
        expected = {index: tensor.clone() for index, tensor in recorded.items()}
        actual = BoundaryWrapper(native)(ids, mask)
    assert len(actual) == 5
    torch.testing.assert_close(actual[0], final, rtol=0, atol=0)
    for index, tensor in zip(BOUNDARIES, actual[1:], strict=True):
        torch.testing.assert_close(tensor, expected[index], rtol=0, atol=0)
