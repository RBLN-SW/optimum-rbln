from types import SimpleNamespace

import torch
from transformers import ModernBertConfig, ModernBertModel
from transformers.modeling_outputs import BaseModelOutput

from optimum.rbln import RBLNAutoConfig, RBLNAutoModel, RBLNModernBertModel, RBLNModernBertModelConfig


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


def test_base_wrapper_and_public_outputs_match_native():
    native = tiny_model()
    wrapper = RBLNModernBertModel._wrap_model_if_needed(native, RBLNModernBertModelConfig())
    runtime = object.__new__(RBLNModernBertModel)
    runtime.config = native.config
    runtime.rbln_config = SimpleNamespace(compile_cfgs=[SimpleNamespace(is_multiple_input_info=False)])
    runtime.model = [wrapper]
    ids = torch.randint(1, 128, (1, 17), generator=torch.Generator().manual_seed(42))
    mask = torch.ones_like(ids)
    mask[:, -3:] = 0
    with torch.inference_mode():
        expected = native(input_ids=ids, attention_mask=mask).last_hidden_state
        for return_dict in (False, True):
            actual = runtime(input_ids=ids, attention_mask=mask, return_dict=return_dict)
            assert isinstance(actual, BaseModelOutput if return_dict else tuple)
            assert len(actual) == 1
            torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)


def test_sdpa_selection(monkeypatch):
    from optimum.rbln.transformers.modeling_generic import RBLNTransformerEncoderForFeatureExtraction

    calls = []
    monkeypatch.setattr(
        RBLNTransformerEncoderForFeatureExtraction,
        "get_pytorch_model",
        classmethod(lambda cls, *args, **kwargs: calls.append(kwargs) or object()),
    )
    RBLNModernBertModel.get_pytorch_model("model", attn_implementation="flash_attention_2")
    assert calls[0]["attn_implementation"] == "sdpa"
