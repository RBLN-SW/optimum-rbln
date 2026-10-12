import pytest
import torch
from transformers import (
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    DebertaV2ForQuestionAnswering,
    DebertaV2ForSequenceClassification,
    DebertaV2Model,
)


@pytest.mark.parametrize(
    "model_class",
    [DebertaV2Model, DebertaV2ForMaskedLM, DebertaV2ForSequenceClassification, DebertaV2ForQuestionAnswering],
)
@pytest.mark.parametrize("type_vocab_size", [0, 2])
def test_deberta_v2_wrapper_preserves_relative_attention_and_padding(model_class, type_vocab_size):
    import optimum.rbln as rbln

    torch.manual_seed(42)
    model = model_class(
        DebertaV2Config(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
            relative_attention=True,
            pos_att_type=["c2p", "p2c"],
            position_buckets=16,
            conv_kernel_size=3,
            type_vocab_size=type_vocab_size,
        )
    ).eval()
    input_ids = torch.randint(1, 64, (2, 17))
    input_ids[:, -3:] = 0
    attention_mask = input_ids.ne(0).long()
    token_type_ids = torch.zeros_like(input_ids)
    token_type_ids[:, 5:10] = 1
    rbln_model_class = getattr(rbln, f"RBLN{model_class.__name__}")
    config_class = getattr(rbln, f"RBLN{model_class.__name__}Config")
    config = rbln_model_class._update_rbln_config(
        None, model, model.config, config_class(max_seq_len=17, batch_size=2)
    )
    wrapper = rbln_model_class._wrap_model_if_needed(model, config)

    with torch.no_grad():
        expected = model(input_ids, attention_mask, token_type_ids, return_dict=False)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        actual = wrapper(**{name: inputs[name] for name in config.model_input_names})

    expected_names = ["input_ids", "attention_mask"]
    if type_vocab_size:
        expected_names.append("token_type_ids")
    assert config.model_input_names == expected_names
    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@pytest.mark.parametrize("type_vocab_size", [0, 2])
@pytest.mark.parametrize("with_attention_mask", [False, True])
@pytest.mark.parametrize("with_token_type_ids", [False, True])
def test_deberta_v2_runtime_accepts_optional_masks(type_vocab_size, with_attention_mask, with_token_type_ids):
    from optimum.rbln import RBLNDebertaV2Model, RBLNDebertaV2ModelConfig

    torch.manual_seed(42)
    reference = DebertaV2Model(
        DebertaV2Config(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
            type_vocab_size=type_vocab_size,
        )
    ).eval()
    config = RBLNDebertaV2Model._update_rbln_config(
        None, reference, reference.config, RBLNDebertaV2ModelConfig(max_seq_len=17, batch_size=2)
    )
    model = object.__new__(RBLNDebertaV2Model)
    model.config = reference.config
    model.rbln_config = config
    model.model = [RBLNDebertaV2Model._wrap_model_if_needed(reference, config)]
    input_ids = torch.randint(1, 64, (2, 17))
    mask = torch.ones_like(input_ids)
    mask[:, -3:] = 0
    token_types = torch.zeros_like(input_ids)
    token_types[:, 5:10] = 1
    kwargs = {}
    if with_attention_mask:
        kwargs["attention_mask"] = mask
    if with_token_type_ids:
        kwargs["token_type_ids"] = token_types

    with torch.no_grad():
        expected = reference(input_ids, return_dict=False, **kwargs)
        actual = model(input_ids, return_dict=False, **kwargs)

    torch.testing.assert_close(actual[0], expected[0])
