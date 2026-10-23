import pytest
import torch
from transformers import MPNetConfig, MPNetModel


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("sequence_length", [8, 17])
@pytest.mark.parametrize("add_pooling_layer", [True, False])
def test_mpnet_wrapper_preserves_padded_outputs(batch_size, sequence_length, add_pooling_layer):
    from optimum.rbln import RBLNMPNetModel, RBLNMPNetModelConfig

    torch.manual_seed(42)
    model = MPNetModel(
        MPNetConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=64,
        ),
        add_pooling_layer=add_pooling_layer,
    ).eval()
    input_ids = torch.randint(4, 64, (batch_size, sequence_length))
    input_ids[:, -3:] = model.config.pad_token_id
    attention_mask = input_ids.ne(model.config.pad_token_id).long()
    wrapper = RBLNMPNetModel._wrap_model_if_needed(
        model, RBLNMPNetModelConfig(max_seq_len=sequence_length, batch_size=batch_size)
    )

    with torch.no_grad():
        expected = tuple(t for t in model(input_ids, attention_mask, return_dict=False) if t is not None)
        actual = wrapper(input_ids, attention_mask)

    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@pytest.mark.parametrize("sequence_length", [None, 62, [8, 62]])
def test_mpnet_position_limit_excludes_reserved_positions(sequence_length):
    from optimum.rbln import RBLNMPNetModel, RBLNMPNetModelConfig

    model = MPNetModel(
        MPNetConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=64,
        )
    ).eval()
    config = RBLNMPNetModel._update_rbln_config(
        None, model, model.config, RBLNMPNetModelConfig(max_seq_len=sequence_length)
    )
    assert config.max_seq_len == (62 if sequence_length is None else sequence_length)


@pytest.mark.parametrize("sequence_length", [63, [8, 63]])
def test_mpnet_rejects_sequences_beyond_position_limit(sequence_length):
    from optimum.rbln import RBLNMPNetModel, RBLNMPNetModelConfig

    model = MPNetModel(
        MPNetConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=64,
        )
    ).eval()
    with pytest.raises(ValueError, match="max_seq_len"):
        RBLNMPNetModel._update_rbln_config(
            None, model, model.config, RBLNMPNetModelConfig(max_seq_len=sequence_length)
        )


def test_mpnet_wrapper_uses_default_attention_mask():
    from optimum.rbln import RBLNMPNetModel, RBLNMPNetModelConfig

    model = MPNetModel(
        MPNetConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64)
    ).eval()
    input_ids = torch.randint(4, 64, (1, 8))
    wrapper = RBLNMPNetModel._wrap_model_if_needed(model, RBLNMPNetModelConfig(max_seq_len=8))

    with torch.no_grad():
        expected = model(input_ids=input_ids, return_dict=False)
        actual = wrapper(input_ids=input_ids)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@pytest.mark.parametrize("add_pooling_layer", [True, False])
def test_mpnet_exports_tensor_outputs_without_pooler(add_pooling_layer):
    import rebel

    from optimum.rbln import RBLNMPNetModel, RBLNMPNetModelConfig

    model = MPNetModel(
        MPNetConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64),
        add_pooling_layer=add_pooling_layer,
    ).eval()
    wrapper = RBLNMPNetModel._wrap_model_if_needed(model, RBLNMPNetModelConfig(max_seq_len=8))
    inputs = (torch.randint(4, 64, (1, 8)), torch.ones((1, 8), dtype=torch.int64))
    rebel.torch_to_ir(
        wrapper,
        input_info=[("input_ids", [1, 8], "int64"), ("attention_mask", [1, 8], "int64")],
        example_inputs=inputs,
    )


@pytest.mark.parametrize("return_dict", [False, True])
def test_mpnet_without_pooler_preserves_runtime_output_contract(return_dict):
    from optimum.rbln import RBLNMPNetModel

    model = object.__new__(RBLNMPNetModel)
    hidden = torch.randn(1, 8, 32)
    output = model._prepare_output(hidden, return_dict)
    if return_dict:
        assert output.last_hidden_state is hidden
        assert output.pooler_output is None
    else:
        assert output == (hidden, None)
