import pytest
import torch
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification, DebertaV2Model

from optimum.rbln.transformers.models.deberta_v2 import (
    RBLNDebertaV2ForSequenceClassification,
    RBLNDebertaV2ForSequenceClassificationConfig,
    RBLNDebertaV2Model,
    RBLNDebertaV2ModelConfig,
)


def tiny_config(type_vocab_size=0, conv_kernel_size=0, position_buckets=16):
    return DebertaV2Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        relative_attention=True,
        position_buckets=position_buckets,
        max_relative_positions=64,
        pos_att_type=["c2p", "p2c"],
        share_att_key=True,
        position_biased_input=False,
        norm_rel_ebd="layer_norm",
        type_vocab_size=type_vocab_size,
        conv_kernel_size=conv_kernel_size,
        num_labels=32,
    )


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("type_vocab_size", [0, 2])
@pytest.mark.parametrize("conv_kernel_size", [0, 3])
@pytest.mark.parametrize("position_buckets", [-1, 16])
def test_native_relative_attention_and_optional_v2_convolution_parity(
    classification, type_vocab_size, conv_kernel_size, position_buckets
):
    torch.manual_seed(42)
    hf_class = DebertaV2ForSequenceClassification if classification else DebertaV2Model
    rbln_class = RBLNDebertaV2ForSequenceClassification if classification else RBLNDebertaV2Model
    model = hf_class(tiny_config(type_vocab_size, conv_kernel_size, position_buckets)).eval()
    wrapper = rbln_class._wrap_model_if_needed(model, None)
    ids = torch.randint(1, 64, (2, 17))
    mask = torch.ones_like(ids)
    mask[:, -3:] = 0
    types = torch.zeros_like(ids)
    if type_vocab_size:
        types[:, 5:10] = 1
    inputs = (ids, mask, types) if type_vocab_size else (ids, mask)
    with torch.no_grad():
        expected = model(*inputs, return_dict=False)
        actual = wrapper(*inputs)
        traced = torch.jit.trace(wrapper, inputs)
        traced_actual = traced(*inputs)
    for output, reference, traced_output in zip(actual, expected, traced_actual, strict=True):
        torch.testing.assert_close(output, reference)
        torch.testing.assert_close(traced_output, reference)
    assert wrapper.model is not model
    assert next(wrapper.parameters()).data_ptr() != next(model.parameters()).data_ptr()
    assert model.config.return_dict is True


@pytest.mark.parametrize("type_vocab_size", [0, 2])
@pytest.mark.parametrize("explicit_names", [False, True])
def test_config_omits_unused_token_types_even_when_tokenizer_advertises_them(type_vocab_size, explicit_names):
    model = DebertaV2Model(tiny_config(type_vocab_size)).eval()
    names = ["input_ids", "attention_mask", "token_type_ids"]
    tokenizer = type("Tokenizer", (), {"model_input_names": names})()
    config = RBLNDebertaV2ModelConfig(max_seq_len=[8, 32], model_input_names=names if explicit_names else None)
    updated = RBLNDebertaV2Model._update_rbln_config([tokenizer], model, model.config, config)
    assert updated.model_input_names == (names if type_vocab_size else names[:2])
    assert updated.compile_cfgs[0].is_multiple_input_info
    for info in updated.compile_cfgs[0].input_info:
        assert [entry[0] for entry in info] == updated.model_input_names


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("return_dict", [False, True])
@pytest.mark.parametrize("original_padding", [False, True])
def test_bucketed_outputs_preserve_native_valid_tokens(classification, return_dict, original_padding):
    torch.manual_seed(42)
    hf_class = DebertaV2ForSequenceClassification if classification else DebertaV2Model
    rbln_class = RBLNDebertaV2ForSequenceClassification if classification else RBLNDebertaV2Model
    config_class = RBLNDebertaV2ForSequenceClassificationConfig if classification else RBLNDebertaV2ModelConfig
    native = hf_class(tiny_config()).eval()
    config = rbln_class._update_rbln_config(None, native, native.config, config_class(max_seq_len=[8, 32]))
    model = object.__new__(rbln_class)
    model.config = native.config
    model.rbln_config = config
    model.model = [rbln_class._wrap_model_if_needed(native, config)]
    ids = torch.randint(1, 64, (1, 17))
    mask = torch.ones_like(ids)
    if original_padding:
        mask[:, -3:] = 0
    with torch.no_grad():
        expected = native(ids, mask, return_dict=return_dict)
        actual = model.forward(input_ids=ids, attention_mask=mask, return_dict=return_dict)
        padded_expected = native(
            torch.nn.functional.pad(ids, (0, 15)), torch.nn.functional.pad(mask, (0, 15)), return_dict=False
        )
    if return_dict:
        assert type(actual) is type(expected)
        actual, expected = actual.to_tuple(), expected.to_tuple()
    for output, reference in zip(actual, expected, strict=True):
        assert output.shape == reference.shape
        if classification:
            torch.testing.assert_close(output, reference)
        else:
            torch.testing.assert_close(output[:, mask[0].bool()], reference[:, mask[0].bool()])
            torch.testing.assert_close(output, padded_expected[0][:, :17])
            if original_padding:
                # Native fully masked queries softmax to a uniform row; their
                # ignored outputs depend on length. Valid tokens stay exact.
                assert (reference[:, -3:] - output[:, -3:]).abs().max() > 1e-4


@pytest.mark.parametrize("classification", [False, True])
def test_wrapper_defaults_optional_inputs(classification):
    hf_class = DebertaV2ForSequenceClassification if classification else DebertaV2Model
    rbln_class = RBLNDebertaV2ForSequenceClassification if classification else RBLNDebertaV2Model
    native = hf_class(tiny_config()).eval()
    wrapper = rbln_class._wrap_model_if_needed(native, None)
    ids = torch.ones((1, 8), dtype=torch.long)
    with torch.no_grad():
        torch.testing.assert_close(wrapper(ids)[0], native(ids, return_dict=False)[0])


@pytest.mark.parametrize("pos_att_type", [["c2p"], ["p2c"], ["c2p", "p2c"]])
def test_v2_separate_relative_projections_and_absolute_positions(pos_att_type):
    config = tiny_config(type_vocab_size=2, conv_kernel_size=3)
    config.share_att_key = False
    config.position_biased_input = True
    config.pos_att_type = pos_att_type
    native = DebertaV2Model(config).eval()
    wrapper = RBLNDebertaV2Model._wrap_model_if_needed(native, None)
    ids = torch.randint(1, 64, (1, 17))
    mask = torch.ones_like(ids)
    mask[:, -3:] = 0
    types = torch.zeros_like(ids)
    types[:, 6:] = 1
    with torch.no_grad():
        torch.testing.assert_close(wrapper(ids, mask, types)[0], native(ids, mask, types).last_hidden_state)


@pytest.mark.parametrize("classification", [False, True])
def test_export_disables_diagnostic_outputs_without_mutating_source(classification):
    hf_class = DebertaV2ForSequenceClassification if classification else DebertaV2Model
    rbln_class = RBLNDebertaV2ForSequenceClassification if classification else RBLNDebertaV2Model
    config = tiny_config()
    config.output_hidden_states = True
    config.output_attentions = True
    native = hf_class(config).eval()
    outputs = rbln_class._wrap_model_if_needed(native, None)(torch.ones((1, 8), dtype=torch.long))
    assert len(outputs) == 1
    assert native.config.output_hidden_states
    assert native.config.output_attentions


def test_runtime_defaults_mask_and_discards_unused_token_types():
    native = DebertaV2Model(tiny_config()).eval()
    config = RBLNDebertaV2Model._update_rbln_config(
        None, native, native.config, RBLNDebertaV2ModelConfig(max_seq_len=8)
    )
    model = object.__new__(RBLNDebertaV2Model)
    model.config, model.rbln_config = native.config, config

    def strict_runtime(input_ids, attention_mask):
        return native(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

    model.model = [strict_runtime]
    ids = torch.randint(1, 64, (1, 8))
    with torch.no_grad():
        actual = model.forward(input_ids=ids, token_type_ids=torch.ones_like(ids)).last_hidden_state
        torch.testing.assert_close(actual, native(ids).last_hidden_state)
