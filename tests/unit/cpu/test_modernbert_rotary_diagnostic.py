import copy

import pytest
import torch

from scripts.diagnostics.modernbert_rotary import install_fixed_rotary

from .test_modernbert import tiny_model


@pytest.mark.parametrize("length", [17, 8192])
def test_fixed_default_position_rotary_matches_both_layer_types_exactly(length):
    original = tiny_model()
    candidate = install_fixed_rotary(copy.deepcopy(original))
    assert next(iter(candidate.named_parameters()))[0] == next(iter(original.named_parameters()))[0]
    for name, value in original.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)
    x = torch.empty(1, length, original.config.hidden_size)
    ids = torch.arange(length)[None]
    for layer_type in original.rotary_emb.layer_types:
        for got, expected in zip(
            candidate.rotary_emb(x, ids, layer_type), original.rotary_emb(x, ids, layer_type), strict=True
        ):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
    inputs = {
        "input_ids": torch.randint(1, 128, (1, length)),
        "attention_mask": torch.ones(1, length, dtype=torch.int64),
    }
    inputs["attention_mask"][:, -3:] = 0
    with torch.inference_mode():
        torch.testing.assert_close(
            candidate(**inputs).last_hidden_state, original(**inputs).last_hidden_state, rtol=0, atol=0
        )


def test_capture_raw_pair_is_saved_before_metric_gate(tmp_path, monkeypatch):
    from transformers.modeling_outputs import BaseModelOutput

    from scripts.diagnostics.modernbert_capture import Capture

    inputs = {"input_ids": torch.arange(8)[None], "attention_mask": torch.ones(1, 8, dtype=torch.int64)}
    expected = torch.randn(1, 8, 4)
    actual = expected + 0.5
    capture = Capture(tmp_path)
    capture.reference(inputs, BaseModelOutput(last_hidden_state=expected))
    monkeypatch.setattr(
        torch.nn.functional,
        "cosine_similarity",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("metric failed")),
    )
    with pytest.raises(RuntimeError, match="metric failed"):
        capture.actual(inputs, BaseModelOutput(last_hidden_state=actual))
    raw = torch.load(tmp_path / "offset-0.pt", weights_only=True)
    assert torch.equal(raw["actual"], actual)
    assert torch.equal(raw["reference"], expected)
    assert torch.equal(raw["inputs"]["input_ids"], inputs["input_ids"])


def test_capture_rejects_changed_output_with_original_threshold(tmp_path):
    from transformers.modeling_outputs import BaseModelOutput

    from scripts.diagnostics.modernbert_capture import Capture

    inputs = {"input_ids": torch.arange(8)[None], "attention_mask": torch.ones(1, 8, dtype=torch.int64)}
    expected = torch.randn(1, 8, 4)
    capture = Capture(tmp_path)
    capture.reference(inputs, BaseModelOutput(last_hidden_state=expected))
    capture.actual(inputs, BaseModelOutput(last_hidden_state=expected + 10))
    assert capture.rows[0]["threshold_passed"] is False
    assert capture.rows[0]["nrmse"] > 0.05


@pytest.mark.parametrize("variable", ["SAVE_ARTIFACTS_PATH", "REUSE_ARTIFACTS_PATH"])
def test_capture_guard_runs_before_any_npu_or_tests(tmp_path, monkeypatch, variable):
    import json

    from scripts.diagnostics import modernbert_capture as capture

    monkeypatch.setenv(variable, "compat-artifact")
    monkeypatch.setattr(capture, "run", lambda *args: pytest.fail("runtime invoked"))
    assert capture.main(["--execute", "--output", str(tmp_path)]) == 1
    result = json.loads((tmp_path / "original/result.json").read_text())
    assert result["accepted"] is False
    assert "compatibility mode" in result["error"]


def test_original_failure_is_preserved_after_all_four_raw_pairs_and_state_checks(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import transformers
    from transformers.modeling_outputs import BaseModelOutput

    from optimum.rbln import RBLNModernBertModel
    from scripts.diagnostics.modernbert_capture import run
    from tests import test_transformers

    case = test_transformers.TestModernBertModelNativeMaxSeqLen

    def tensor(inputs):
        return (inputs["input_ids"][..., None].float() + torch.arange(768) / 768) / 8192

    class Reference:
        def eval(self):
            return self

        def __call__(self, **inputs):
            return self.forward(**inputs)

        def forward(self, **inputs):
            return BaseModelOutput(last_hidden_state=tensor(inputs))

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            offset = case.TEXTS.index(texts[0])
            ids = (torch.arange(8192)[None] + offset) % 127
            mask = torch.zeros_like(ids)
            mask[:, : [12, 19, 8192, 7][offset]] = 1
            return transformers.BatchEncoding({"input_ids": ids, "attention_mask": mask})

    def runtime():
        result = object.__new__(RBLNModernBertModel)
        result.config = SimpleNamespace(return_dict=True)
        result.rbln_config = SimpleNamespace(compile_cfgs=[SimpleNamespace(is_multiple_input_info=False)])

        def forward(**inputs):
            output = tensor(inputs)
            if inputs["attention_mask"].sum() == 8192:
                output += 0.1
            return (output,)

        result.model = [forward]
        return result

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: (Reference(), {"missing_keys": [], "unexpected_keys": []}),
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(case, "RBLN_AUTO_CLASS", SimpleNamespace(from_pretrained=lambda *args, **kwargs: runtime()))
    monkeypatch.setattr(case, "get_rbln_local_dir", classmethod(lambda cls: str(tmp_path / "model")))
    monkeypatch.setattr(case, "model", runtime(), raising=False)

    def pytest_main(args, plugins):
        capture = plugins[0]
        capture.pytest_collection_finish(SimpleNamespace(items=list(range(9))))
        with pytest.raises(AssertionError):
            case("test_embedding_parity").test_embedding_parity()
        capture.failed.append("original_numerical_gate")
        return 1

    monkeypatch.setattr(pytest, "main", pytest_main)
    code, capture = run(tmp_path, "original")
    assert code == 1
    assert len(capture.rows) == 4
    assert capture.rows[2]["threshold_passed"] is False
    assert not capture.supplement_errors
    assert len(list(tmp_path.glob("offset-*.pt"))) == 4
    assert "repeat_tuple_return_reload" in (tmp_path / "calls.jsonl").read_text()
