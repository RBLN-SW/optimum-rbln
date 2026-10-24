from types import SimpleNamespace

import pytest
import torch
from transformers import ModernBertConfig, ModernBertForMaskedLM, ModernBertModel
from transformers.modeling_outputs import BaseModelOutput

from optimum.rbln import RBLNAutoConfig, RBLNAutoModel, RBLNModernBertModel, RBLNModernBertModelConfig
from optimum.rbln.transformers.models.modernbert.modernbert_architecture import ModernBertModelWrapper


def tiny_model():
    config = ModernBertConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=8192,
        local_attention=8,
        global_attn_every_n_layers=3,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
        architectures=["ModernBertModel"],
        reference_compile=False,
    )
    config._attn_implementation = "sdpa"
    return ModernBertModel(config).eval()


def test_auto_discovery_and_saved_config(tmp_path):
    native = tiny_model()
    native.config.save_pretrained(tmp_path)
    assert RBLNAutoModel.get_rbln_cls(tmp_path, export=True) is RBLNModernBertModel
    config = RBLNModernBertModel._update_rbln_config(
        None, native, native.config, RBLNModernBertModelConfig(batch_size=1)
    )
    assert config.compile_cfgs[0].input_info == [
        ("input_ids", [1, 8192], "int64"),
        ("attention_mask", [1, 8192], "int64"),
    ]
    config.save(tmp_path)
    assert isinstance(RBLNAutoConfig.from_pretrained(tmp_path), RBLNModernBertModelConfig)


@pytest.mark.parametrize("return_dict", [False, True])
@pytest.mark.parametrize("length", [17, 8192])
def test_base_wrapper_and_public_outputs_match_native(return_dict, length):
    native = tiny_model()
    wrapper = RBLNModernBertModel._wrap_model_if_needed(native, RBLNModernBertModelConfig())
    runtime = object.__new__(RBLNModernBertModel)
    runtime.config = native.config
    runtime.rbln_config = SimpleNamespace(compile_cfgs=[SimpleNamespace(is_multiple_input_info=False)])
    runtime.model = [wrapper]
    ids = torch.randint(1, 128, (1, length), generator=torch.Generator().manual_seed(42))
    mask = torch.ones_like(ids)
    mask[:, -3:] = 0
    with torch.inference_mode():
        expected = native(input_ids=ids, attention_mask=mask).last_hidden_state
        actual = runtime(input_ids=ids, attention_mask=mask, return_dict=return_dict)
    assert isinstance(actual, BaseModelOutput if return_dict else tuple)
    assert len(actual) == 1
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    assert torch.isfinite(actual[0]).all()
    pooled = (actual[0] * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    reference = (expected * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    torch.testing.assert_close(torch.nn.functional.normalize(pooled), torch.nn.functional.normalize(reference))


def test_sdpa_selection_and_existing_mlm_wrapper(monkeypatch):
    from optimum.rbln.transformers.modeling_generic import RBLNTransformerEncoderForFeatureExtraction

    calls = []
    monkeypatch.setattr(
        RBLNTransformerEncoderForFeatureExtraction,
        "get_pytorch_model",
        classmethod(lambda cls, *args, **kwargs: calls.append(kwargs) or object()),
    )
    RBLNModernBertModel.get_pytorch_model("model", attn_implementation="flash_attention_2")
    assert calls[0]["attn_implementation"] == "sdpa"
    native = ModernBertForMaskedLM(tiny_model().config).eval()
    inputs = {
        "input_ids": torch.ones(1, 17, dtype=torch.int64),
        "attention_mask": torch.ones(1, 17, dtype=torch.int64),
    }
    with torch.inference_mode():
        torch.testing.assert_close(ModernBertModelWrapper(native, None)(**inputs)[0], native(**inputs).logits)


@pytest.mark.parametrize("failure", [None, "numerical", "return_to_first", "reload_first", "reload_last"])
def test_persistent_native_parity_checks_all_four_texts_and_rejects_corruption(monkeypatch, failure):
    import transformers

    from tests.test_transformers import TestModernBertModelNativeMaxSeqLen

    seen = []

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            from transformers import BatchEncoding

            seen.extend(texts)
            assert kwargs["max_length"] == 8192
            assert kwargs["padding"] == "max_length"
            ids = (torch.arange(8192).reshape(1, -1) + Case.TEXTS.index(texts[0])) % 17
            mask = torch.ones_like(ids)
            if len(texts[0]) < 100:
                mask[:, 13:] = 0
            return BatchEncoding({"input_ids": ids, "attention_mask": mask})

    class Model:
        def __init__(self, reloaded=False):
            self.calls = 0
            self.reloaded = reloaded

        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask, return_dict=True):
            hidden = (input_ids[..., None].float() + torch.arange(768).float() / 768) / 17
            self.calls += 1
            if (
                (failure == "numerical" and self is Case.model)
                or (failure == "return_to_first" and self is Case.model and self.calls == 13)
                or (failure == "reload_first" and self.reloaded and self.calls == 1)
                or (failure == "reload_last" and self.reloaded and self.calls == 2)
            ):
                hidden[:, :13] += 100
            output = BaseModelOutput(last_hidden_state=hidden)
            return output if return_dict else output.to_tuple()

    class Case(TestModernBertModelNativeMaxSeqLen):
        model = Model()
        RBLN_AUTO_CLASS = SimpleNamespace(from_pretrained=lambda *args, **kwargs: Model(reloaded=True))
        get_rbln_local_dir = classmethod(lambda cls: "local-artifact")

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: (Model(), {"missing_keys": [], "unexpected_keys": []}),
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    case = Case("test_embedding_parity")
    if failure:
        with pytest.raises(AssertionError):
            case.test_embedding_parity()
    else:
        case.test_embedding_parity()
        assert seen == Case.TEXTS
