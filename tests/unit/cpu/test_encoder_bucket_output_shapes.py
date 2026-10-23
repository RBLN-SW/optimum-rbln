import pytest
import torch
from optimum import rbln
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForQuestionAnswering,
    BertModel,
    RobertaConfig,
    RobertaForSequenceClassification,
)


@pytest.mark.parametrize("return_dict", [False, True])
@pytest.mark.parametrize("kind", ["pooler", "classification", "masked_lm", "qa"])
def test_bucketing_unpads_only_sequence_outputs(kind, return_dict):
    torch.manual_seed(42)
    config = {
        "vocab_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "max_position_embeddings": 64,
        "num_labels": 32,
    }
    if kind == "classification":
        native = RobertaForSequenceClassification(RobertaConfig(**config)).eval()
        model_class = rbln.RBLNRobertaForSequenceClassification
        config_class = rbln.RBLNRobertaForSequenceClassificationConfig
    else:
        hf_class, model_class, config_class = {
            "pooler": (BertModel, rbln.RBLNBertModel, rbln.RBLNBertModelConfig),
            "masked_lm": (BertForMaskedLM, rbln.RBLNBertForMaskedLM, rbln.RBLNBertForMaskedLMConfig),
            "qa": (
                BertForQuestionAnswering,
                rbln.RBLNBertForQuestionAnswering,
                rbln.RBLNBertForQuestionAnsweringConfig,
            ),
        }[kind]
        if kind == "qa":
            config["num_labels"] = 2
        native = hf_class(BertConfig(**config)).eval()
    updated = model_class._update_rbln_config(None, native, native.config, config_class(max_seq_len=[8, 32]))
    model = object.__new__(model_class)
    model.config, model.rbln_config = native.config, updated

    def runtime(*args, **kwargs):
        return native(*args, **kwargs, return_dict=False)

    model.model = [runtime]
    ids = torch.randint(4, 64, (1, 17))
    mask = torch.ones_like(ids)
    mask[:, -3:] = 0
    with torch.no_grad():
        expected = native(ids, mask, return_dict=return_dict)
        actual = model.forward(input_ids=ids, attention_mask=mask, return_dict=return_dict)
    if return_dict:
        assert type(actual) is type(expected)
        expected, actual = expected.to_tuple(), actual.to_tuple()
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        assert a.shape == e.shape
        torch.testing.assert_close(a, e)


def test_bucketing_rejects_compile_interface_without_attention_mask():
    native = BertModel(BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64))
    config = rbln.RBLNBertModelConfig(max_seq_len=[8, 32], model_input_names=["input_ids"])
    with pytest.raises(ValueError, match="attention_mask"):
        rbln.RBLNBertModel._update_rbln_config(None, native, native.config, config)


@pytest.mark.parametrize("model_name", ["Bart", "Pegasus"])
def test_seq2seq_compact_tuple_unpads_both_hidden_outputs(model_name):
    import transformers

    config = getattr(transformers, f"{model_name}Config")(
        vocab_size=64,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_position_embeddings=64,
    )
    native = getattr(transformers, f"{model_name}Model")(config).eval()
    model_class = getattr(rbln, f"RBLN{model_name}Model")
    config_class = getattr(rbln, f"RBLN{model_name}ModelConfig")
    updated = model_class._update_rbln_config(None, native, config, config_class(max_seq_len=[8, 32]))
    model = object.__new__(model_class)
    model.config, model.rbln_config = config, updated

    def runtime(*args, **kwargs):
        return native(*args, **kwargs, use_cache=False, return_dict=False)

    model.model = [runtime]
    inputs = {name: torch.ones((1, 17), dtype=torch.long) for name in updated.model_input_names}
    inputs["input_ids"] = torch.randint(4, 64, (1, 17))
    if "decoder_input_ids" in inputs:
        inputs["decoder_input_ids"] = inputs["input_ids"].clone()
    with torch.no_grad():
        expected = native(**inputs, use_cache=False, return_dict=False)
        actual = model.forward(**inputs, return_dict=False)
    assert len(actual) == len(expected) == 2
    for a, e in zip(actual, expected, strict=True):
        assert a.shape == e.shape
        torch.testing.assert_close(a, e)
