import pytest
import torch
from transformers import MPNetConfig, MPNetModel


@pytest.mark.parametrize("add_pooling_layer", [True, False])
def test_mpnet_wrapper_preserves_position_bias_and_padded_outputs(add_pooling_layer):
    from optimum.rbln import RBLNMPNetModel, RBLNMPNetModelConfig

    torch.manual_seed(42)
    model = MPNetModel(
        MPNetConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64),
        add_pooling_layer=add_pooling_layer,
    ).eval()
    input_ids = torch.randint(4, 64, (4, 17))
    mask = torch.arange(17)[None, :] < torch.tensor([2, 7, 13, 17])[:, None]
    input_ids[~mask] = model.config.pad_token_id
    hidden = torch.zeros(4, 17, model.config.hidden_size)
    with torch.no_grad():
        original_bias = model.encoder.compute_position_bias(hidden)
        expected = tuple(t for t in model(input_ids, mask.long(), return_dict=False) if t is not None)
        wrapper = RBLNMPNetModel._wrap_model_if_needed(model, RBLNMPNetModelConfig(max_seq_len=17, batch_size=4))
        shared_bias = wrapper.model.encoder.compute_position_bias(hidden)
        actual = wrapper(input_ids, mask.long())
        reused = RBLNMPNetModel._wrap_model_if_needed(model, RBLNMPNetModelConfig(max_seq_len=17, batch_size=4))
        for first, second in zip(actual, reused(input_ids, mask.long()), strict=True):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    assert wrapper.model is model
    assert shared_bias.shape == (1, model.config.num_attention_heads, 17, 17)
    torch.testing.assert_close(shared_bias.expand_as(original_bias), original_bias, rtol=0, atol=0)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@pytest.mark.parametrize("sequence_length", [None, [8, 62]])
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
    with torch.no_grad():
        expected = model(input_ids=input_ids, return_dict=False)
        wrapper = RBLNMPNetModel._wrap_model_if_needed(model, RBLNMPNetModelConfig(max_seq_len=8))
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


@pytest.mark.parametrize("option", ["output_hidden_states", "output_attentions"])
def test_mpnet_rejects_unsupported_output_options_before_runtime(option):
    from unittest.mock import patch

    from optimum.rbln import RBLNMPNetModel
    from optimum.rbln.transformers.modeling_generic import RBLNTransformerEncoderForFeatureExtraction

    model = object.__new__(RBLNMPNetModel)
    model.config = MPNetConfig(**{option: True})
    with patch.object(RBLNTransformerEncoderForFeatureExtraction, "forward", return_value=()) as runtime:
        for options in ({}, {option: True}):
            with pytest.raises(NotImplementedError, match=option):
                model(input_ids=torch.ones((1, 8), dtype=torch.int64), **options)
        runtime.assert_not_called()
        assert model(input_ids=torch.ones((1, 8), dtype=torch.int64), **{option: False}) == ()
        runtime.assert_called_once()
