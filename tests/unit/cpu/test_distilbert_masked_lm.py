import pytest
import torch
from transformers import DistilBertConfig, DistilBertForMaskedLM


@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("with_token_types", [False, True])
@pytest.mark.parametrize("return_dict", [False, True])
def test_distilbert_masked_lm_preserves_native_outputs(with_mask, with_token_types, return_dict):
    from optimum.rbln import RBLNDistilBertForMaskedLM, RBLNDistilBertForMaskedLMConfig

    torch.manual_seed(42)
    reference = DistilBertForMaskedLM(
        DistilBertConfig(vocab_size=64, dim=32, hidden_dim=64, n_layers=2, n_heads=4, max_position_embeddings=32)
    ).eval()
    cfg = RBLNDistilBertForMaskedLM._update_rbln_config(
        None, reference, reference.config, RBLNDistilBertForMaskedLMConfig(batch_size=2, max_seq_len=17)
    )
    model = object.__new__(RBLNDistilBertForMaskedLM)
    model.config = reference.config
    model.rbln_config = cfg
    model.model = [RBLNDistilBertForMaskedLM._wrap_model_if_needed(reference, cfg)]
    ids = torch.randint(1, 64, (2, 17))
    mask = torch.ones_like(ids)
    mask[:, -3:] = 0
    kwargs = {"attention_mask": mask} if with_mask else {}
    if with_token_types:
        kwargs["token_type_ids"] = torch.ones_like(ids)
    with torch.no_grad():
        expected = reference(ids, attention_mask=mask if with_mask else None).logits
        output = model(ids, return_dict=return_dict, **kwargs)
    actual = output.logits if return_dict else output[0]
    torch.testing.assert_close(actual, expected)
    assert cfg.model_input_names == ["input_ids", "attention_mask"]


def test_distilbert_masked_lm_auto_dispatch(monkeypatch):
    from optimum.rbln import RBLNAutoModelForMaskedLM, RBLNDistilBertForMaskedLM

    monkeypatch.setattr(RBLNDistilBertForMaskedLM, "from_model", classmethod(lambda cls, model, **kwargs: cls))
    reference = DistilBertForMaskedLM(DistilBertConfig(vocab_size=64, dim=32, hidden_dim=64, n_layers=1, n_heads=4))
    assert RBLNAutoModelForMaskedLM.from_model(reference) is RBLNDistilBertForMaskedLM
