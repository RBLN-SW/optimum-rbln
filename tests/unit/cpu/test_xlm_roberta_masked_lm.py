import pytest
import torch
from transformers import XLMRobertaConfig, XLMRobertaForMaskedLM


@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("return_dict", [False, True])
def test_xlm_roberta_masked_lm_preserves_native_outputs(with_mask, return_dict):
    from optimum.rbln import RBLNXLMRobertaForMaskedLM, RBLNXLMRobertaForMaskedLMConfig

    torch.manual_seed(42)
    reference = XLMRobertaForMaskedLM(
        XLMRobertaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=32,
        )
    ).eval()
    cfg = RBLNXLMRobertaForMaskedLM._update_rbln_config(
        None, reference, reference.config, RBLNXLMRobertaForMaskedLMConfig(batch_size=2, max_seq_len=17)
    )
    model = object.__new__(RBLNXLMRobertaForMaskedLM)
    model.config = reference.config
    model.rbln_config = cfg
    model.model = [RBLNXLMRobertaForMaskedLM._wrap_model_if_needed(reference, cfg)]
    ids = torch.randint(1, 64, (2, 17))
    mask = torch.ones_like(ids)
    mask[:, -3:] = 0
    kwargs = {"attention_mask": mask} if with_mask else {}
    with torch.no_grad():
        expected = reference(ids, attention_mask=mask if with_mask else None).logits
        output = model(ids, return_dict=return_dict, **kwargs)
    actual = output.logits if return_dict else output[0]
    torch.testing.assert_close(actual, expected)
    assert cfg.model_input_names == ["input_ids", "attention_mask"]


def test_xlm_roberta_masked_lm_auto_dispatch(monkeypatch):
    from optimum.rbln import RBLNAutoModelForMaskedLM, RBLNXLMRobertaForMaskedLM

    monkeypatch.setattr(RBLNXLMRobertaForMaskedLM, "from_model", classmethod(lambda cls, model, **kwargs: cls))
    reference = XLMRobertaForMaskedLM(
        XLMRobertaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4
        )
    )
    assert RBLNAutoModelForMaskedLM.from_model(reference) is RBLNXLMRobertaForMaskedLM
