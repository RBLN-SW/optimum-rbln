import pytest
import torch
from transformers import ConvNextConfig, ConvNextForImageClassification, ConvNextModel


@pytest.mark.parametrize("model_class", [ConvNextModel, ConvNextForImageClassification])
@pytest.mark.parametrize("size", [(32, 32), (64, 32)])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("return_dict", [False, True])
def test_convnext_preserves_encoder_and_classifier_outputs(model_class, size, batch, return_dict):
    import optimum.rbln as rbln

    torch.manual_seed(42)
    reference = model_class(ConvNextConfig(depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 32, 64], num_labels=5)).eval()
    cls = getattr(rbln, "RBLN" + model_class.__name__)
    cfg = getattr(rbln, cls.__name__ + "Config")(image_size=size, batch_size=batch)
    cfg = cls._update_rbln_config(None, reference, reference.config, cfg)
    wrapper = cls._wrap_model_if_needed(reference, cfg)
    runtime = object.__new__(cls)
    runtime.config = reference.config
    runtime.rbln_config = cfg
    runtime.model = [wrapper]
    pixels = torch.randn(batch, 3, *size)
    with torch.no_grad():
        expected = reference(pixels, return_dict=False)
        actual = runtime(pixel_values=pixels, return_dict=return_dict)
    if return_dict:
        actual = (
            (actual.logits,)
            if model_class is ConvNextForImageClassification
            else (actual.last_hidden_state, actual.pooler_output)
        )
    assert len(actual) == len(expected)
    for x, y in zip(actual, expected, strict=True):
        torch.testing.assert_close(x, y)
    assert cfg.compile_cfgs[0].input_info == [("pixel_values", [batch, 3, *size], "float32")]


@pytest.mark.parametrize("model_class", [ConvNextModel, ConvNextForImageClassification])
def test_convnext_auto_model_selects_export(model_class, monkeypatch):
    import optimum.rbln as rbln

    cls = getattr(rbln, "RBLN" + model_class.__name__)
    monkeypatch.setattr(cls, "from_model", classmethod(lambda cls, model, **kwargs: cls))
    auto = rbln.RBLNAutoModel if model_class is ConvNextModel else rbln.RBLNAutoModelForImageClassification
    reference = model_class(ConvNextConfig(depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 32, 64]))
    assert auto.from_model(reference) is cls
