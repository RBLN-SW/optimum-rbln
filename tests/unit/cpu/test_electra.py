import pytest
import torch
from transformers import ElectraConfig, ElectraForPreTraining, ElectraForSequenceClassification, ElectraModel


def test_electra_export_exposes_attention_matmuls_without_mutating_native_model():
    from optimum.rbln import RBLNElectraForPreTraining

    model = ElectraForPreTraining(
        ElectraConfig(
            vocab_size=64,
            embedding_size=16,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
        )
    ).eval()
    model.set_attn_implementation("sdpa")
    inputs = (torch.ones((1, 8), dtype=torch.long),) * 3
    wrapper = RBLNElectraForPreTraining._wrap_model_if_needed(model, None)
    traced = torch.jit.trace(wrapper, inputs)

    assert model.config._attn_implementation == "sdpa"
    assert "aten::scaled_dot_product_attention" not in str(traced.inlined_graph)
    with torch.no_grad():
        torch.testing.assert_close(traced(*inputs)[0], model(*inputs).logits)


@pytest.mark.parametrize("model_class", [ElectraModel, ElectraForPreTraining, ElectraForSequenceClassification])
@pytest.mark.parametrize("embedding_size", [16, 32])
def test_electra_wrapper_preserves_projected_embeddings_and_padding(model_class, embedding_size):
    from optimum import rbln

    torch.manual_seed(42)
    model = model_class(
        ElectraConfig(
            vocab_size=64,
            embedding_size=embedding_size,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
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
        actual = wrapper(input_ids, attention_mask, token_type_ids)

    assert config.model_input_names == ["input_ids", "attention_mask", "token_type_ids"]
    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@pytest.mark.parametrize("model_class", [ElectraModel, ElectraForPreTraining, ElectraForSequenceClassification])
@pytest.mark.parametrize("include_mask", [False, True])
def test_electra_wrapper_defaults_optional_inputs(model_class, include_mask):
    from optimum import rbln

    model = model_class(
        ElectraConfig(
            vocab_size=64,
            embedding_size=16,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
        )
    ).eval()
    inputs = {"input_ids": torch.randint(1, 64, (1, 8))}
    if include_mask:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    wrapper = getattr(rbln, f"RBLN{model_class.__name__}")._wrap_model_if_needed(model, None)

    with torch.no_grad():
        expected = model(**inputs, return_dict=False)
        actual = wrapper(**inputs)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_electra_export_disables_optional_diagnostic_outputs():
    from optimum.rbln import RBLNElectraForPreTraining

    model = ElectraForPreTraining(
        ElectraConfig(
            vocab_size=64,
            embedding_size=16,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            output_hidden_states=True,
            output_attentions=True,
        )
    ).eval()
    wrapper = RBLNElectraForPreTraining._wrap_model_if_needed(model, None)
    outputs = wrapper(torch.ones((1, 8), dtype=torch.long))
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 8)
    assert model.config.output_hidden_states
    assert model.config.output_attentions


def test_electra_auto_pretraining_discovery():
    from optimum.rbln import RBLNAutoModelForPreTraining, RBLNElectraForPreTraining

    config = ElectraConfig(architectures=["ElectraForPreTraining"])
    assert RBLNAutoModelForPreTraining.get_rbln_cls("unused", export=True, config=config) is RBLNElectraForPreTraining
