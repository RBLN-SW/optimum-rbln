import pytest
import torch
from transformers import (
    AlbertConfig,
    AlbertForMaskedLM,
    AlbertForQuestionAnswering,
    AlbertForSequenceClassification,
    AlbertModel,
)


@pytest.mark.parametrize(
    "model_class", [AlbertModel, AlbertForMaskedLM, AlbertForSequenceClassification, AlbertForQuestionAnswering]
)
@pytest.mark.parametrize("groups,inner", [(1, 1), (2, 2)])
def test_albert_wrapper_preserves_shared_layers_segments_and_padding(model_class, groups, inner):
    import optimum.rbln as rbln

    torch.manual_seed(42)
    reference = model_class(
        AlbertConfig(
            vocab_size=64,
            embedding_size=16,
            hidden_size=32,
            num_hidden_layers=4,
            num_hidden_groups=groups,
            inner_group_num=inner,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
        )
    ).eval()
    cls = getattr(rbln, f"RBLN{model_class.__name__}")
    config = cls._update_rbln_config(
        None,
        reference,
        reference.config,
        getattr(rbln, f"RBLN{model_class.__name__}Config")(max_seq_len=17, batch_size=2),
    )
    wrapper = cls._wrap_model_if_needed(reference, config)
    ids = torch.randint(1, 64, (2, 17))
    ids[:, -3:] = 0
    inputs = {"input_ids": ids, "attention_mask": ids.ne(0).long(), "token_type_ids": torch.zeros_like(ids)}
    inputs["token_type_ids"][:, 5:10] = 1
    original_storages = [p.data_ptr() for p in reference.parameters()]
    with torch.no_grad():
        expected = reference(**inputs, return_dict=False)
        actual = wrapper(*(inputs[name] for name in config.model_input_names))
    assert [p.data_ptr() for p in wrapper.parameters()] == original_storages
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("with_segments", [False, True])
@pytest.mark.parametrize("return_dict", [False, True])
def test_albert_runtime_optional_inputs_and_pooling(with_mask, with_segments, return_dict):
    from optimum.rbln import RBLNAlbertModel, RBLNAlbertModelConfig

    torch.manual_seed(42)
    reference = AlbertModel(
        AlbertConfig(
            vocab_size=64,
            embedding_size=16,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
        )
    ).eval()
    config = RBLNAlbertModel._update_rbln_config(
        None, reference, reference.config, RBLNAlbertModelConfig(max_seq_len=17, batch_size=2)
    )
    model = object.__new__(RBLNAlbertModel)
    model.config = reference.config
    model.rbln_config = config
    model.model = [RBLNAlbertModel._wrap_model_if_needed(reference, config)]
    ids = torch.randint(1, 64, (2, 17))
    kwargs = {}
    if with_mask:
        kwargs["attention_mask"] = torch.ones_like(ids)
        kwargs["attention_mask"][:, -3:] = 0
    if with_segments:
        kwargs["token_type_ids"] = torch.zeros_like(ids)
        kwargs["token_type_ids"][:, 5:10] = 1
    with torch.no_grad():
        expected = reference(ids, return_dict=return_dict, **kwargs)
        actual = model(ids, return_dict=return_dict, **kwargs)
    if return_dict:
        assert type(actual) is type(expected)
        actual, expected = actual.to_tuple(), expected.to_tuple()
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("custom_positions", [False, True])
def test_albert_custom_position_inputs(custom_positions):
    from optimum.rbln import RBLNAlbertModel, RBLNAlbertModelConfig

    reference = AlbertModel(
        AlbertConfig(
            vocab_size=64,
            embedding_size=16,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
        )
    ).eval()
    config = RBLNAlbertModel._update_rbln_config(
        None,
        reference,
        reference.config,
        RBLNAlbertModelConfig(
            max_seq_len=17,
            batch_size=2,
            model_input_names=["input_ids", "attention_mask", "token_type_ids", "position_ids"],
        ),
    )
    model = object.__new__(RBLNAlbertModel)
    model.config = reference.config
    model.rbln_config = config
    model.model = [RBLNAlbertModel._wrap_model_if_needed(reference, config)]
    ids = torch.randint(1, 64, (2, 17))
    kwargs = {"position_ids": torch.arange(17, 0, -1).expand_as(ids)} if custom_positions else {}
    with torch.no_grad():
        expected = reference(ids, return_dict=False, **kwargs)
        actual = model(ids, return_dict=False, **kwargs)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e)


def test_albert_keeps_segments_when_tokenizer_omits_them():
    from types import SimpleNamespace

    from optimum.rbln import RBLNAlbertModel, RBLNAlbertModelConfig

    reference = AlbertModel(
        AlbertConfig(
            vocab_size=64,
            embedding_size=16,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
        )
    ).eval()
    config = RBLNAlbertModel._update_rbln_config(
        [SimpleNamespace(model_input_names=["input_ids", "attention_mask"])],
        reference,
        reference.config,
        RBLNAlbertModelConfig(max_seq_len=17),
    )
    assert config.model_input_names == ["input_ids", "attention_mask", "token_type_ids"]
