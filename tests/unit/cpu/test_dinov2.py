import pytest
import torch
from transformers import Dinov2Config, Dinov2Model


@pytest.mark.parametrize("size", [(28, 28), (28, 42), (42, 56)])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("return_dict", [False, True])
def test_dinov2_preserves_patch_features_and_pooling(size, batch, return_dict):
    from optimum.rbln import RBLNDinov2Model, RBLNDinov2ModelConfig

    torch.manual_seed(42)
    reference = Dinov2Model(
        Dinov2Config(
            image_size=28,
            patch_size=14,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
        )
    ).eval()
    config = RBLNDinov2Model._update_rbln_config(
        None, reference, reference.config, RBLNDinov2ModelConfig(image_size=size, batch_size=batch)
    )
    wrapper = RBLNDinov2Model._wrap_model_if_needed(reference, config)
    runtime = object.__new__(RBLNDinov2Model)
    runtime.config = reference.config
    runtime.rbln_config = config
    runtime.model = [wrapper]
    pixels = torch.randn(batch, 3, *size)
    with torch.no_grad():
        expected = reference(pixels)
        actual = runtime(pixel_values=pixels, return_dict=return_dict)
    features, pooled = (actual.last_hidden_state, actual.pooler_output) if return_dict else actual
    torch.testing.assert_close(features, expected.last_hidden_state)
    torch.testing.assert_close(pooled, expected.pooler_output)
    assert features.shape == (batch, size[0] // 14 * (size[1] // 14) + 1, 64)
    assert config.compile_cfgs[0].input_info == [("pixel_values", [batch, 3, *size], "float32")]


def test_dinov2_wrapper_exports_only_primary_outputs():
    from optimum.rbln import RBLNDinov2Model, RBLNDinov2ModelConfig

    model = Dinov2Model(
        Dinov2Config(
            image_size=28,
            patch_size=14,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=128,
            output_hidden_states=True,
            output_attentions=True,
        )
    ).eval()
    wrapper = RBLNDinov2Model._wrap_model_if_needed(model, RBLNDinov2ModelConfig(image_size=28))
    with torch.no_grad():
        outputs = wrapper(torch.randn(1, 3, 28, 28))
    assert len(outputs) == 2


def test_dinov2_auto_model_selects_export(monkeypatch):
    from optimum.rbln import RBLNAutoModel, RBLNDinov2Model

    monkeypatch.setattr(RBLNDinov2Model, "from_model", classmethod(lambda cls, model, **kwargs: cls))
    config = Dinov2Config(
        image_size=28, patch_size=14, hidden_size=64, num_hidden_layers=1, num_attention_heads=4, intermediate_size=128
    )
    assert RBLNAutoModel.from_model(Dinov2Model(config)) is RBLNDinov2Model


@pytest.mark.parametrize("explicit_size", [None, 518])
def test_dinov2_compiles_final_crop_size(explicit_size):
    from types import SimpleNamespace

    from optimum.rbln import RBLNDinov2Model, RBLNDinov2ModelConfig

    processor = SimpleNamespace(
        size={"shortest_edge": 256}, crop_size={"height": 224, "width": 224}, do_center_crop=True
    )
    model = Dinov2Model(
        Dinov2Config(image_size=518, patch_size=14, hidden_size=64, num_hidden_layers=1, num_attention_heads=4)
    )
    config = RBLNDinov2Model._update_rbln_config(
        [processor], model, model.config, RBLNDinov2ModelConfig(image_size=explicit_size)
    )
    expected = explicit_size or 224
    assert config.compile_cfgs[0].input_info[0][1] == [1, 3, expected, expected]
