from types import SimpleNamespace

import pytest
import torch
from transformers import Dinov2Config, Dinov2ForImageClassification, Dinov2Model
from transformers.image_utils import SizeDict


def tiny_model(classification=False, **kwargs):
    config = Dinov2Config(
        image_size=28,
        patch_size=14,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        num_labels=5,
        **kwargs,
    )
    return (Dinov2ForImageClassification if classification else Dinov2Model)(config)


def rbln_classes(classification):
    from optimum.rbln import (
        RBLNDinov2ForImageClassification,
        RBLNDinov2ForImageClassificationConfig,
        RBLNDinov2Model,
        RBLNDinov2ModelConfig,
    )

    if classification:
        return RBLNDinov2ForImageClassification, RBLNDinov2ForImageClassificationConfig
    return RBLNDinov2Model, RBLNDinov2ModelConfig


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("size", [(28, 28), (28, 42), (42, 56)])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("return_dict", [False, True])
def test_primary_outputs_match_native(classification, size, batch, return_dict):
    torch.manual_seed(42)
    native = tiny_model(classification).eval()
    cls, config_cls = rbln_classes(classification)
    config = cls._update_rbln_config(None, native, native.config, config_cls(image_size=size, batch_size=batch))
    wrapper = cls._wrap_model_if_needed(native, config)
    runtime = object.__new__(cls)
    runtime.config = native.config
    runtime.rbln_config = config
    runtime.model = [wrapper]
    pixels = torch.randn(batch, 3, *size)
    with torch.no_grad():
        expected = native(pixels)
        actual = runtime(pixel_values=pixels, return_dict=return_dict)
    actual_tuple = actual.to_tuple() if return_dict else actual
    for a, e in zip(actual_tuple, expected.to_tuple(), strict=True):
        torch.testing.assert_close(a, e)
    assert config.compile_cfgs[0].input_info == [("pixel_values", [batch, 3, *size], "float32")]
    if classification:
        assert actual_tuple[0].shape == (batch, 5)
    else:
        assert actual_tuple[0].shape == (batch, size[0] // 14 * (size[1] // 14) + 1, 64)
        torch.testing.assert_close(actual_tuple[1], actual_tuple[0][:, 0])


@pytest.mark.parametrize("classification", [False, True])
def test_wrapper_preserves_source_state_and_fixes_diagnostic_outputs(classification):
    cls, config_cls = rbln_classes(classification)
    native = tiny_model(classification, output_hidden_states=True, output_attentions=True).train()
    attention_implementation = native.config._attn_implementation
    wrapper = cls._wrap_model_if_needed(native, config_cls(image_size=28))
    assert native.training
    assert not wrapper.training
    assert wrapper.model is not native
    assert native.config.output_hidden_states
    assert native.config.output_attentions
    assert native.config._attn_implementation == attention_implementation
    with torch.no_grad():
        output = wrapper(torch.randn(1, 3, 28, 28))
    assert len(output) == (1 if classification else 2)
    assert all(isinstance(value, torch.Tensor) for value in output)


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("explicit_size", [None, 518, (224, 336)])
@pytest.mark.parametrize("center_crop", [False, True])
@pytest.mark.parametrize("crop_type", [dict, SizeDict])
def test_compile_size_follows_final_processor_output(classification, explicit_size, center_crop, crop_type):
    cls, config_cls = rbln_classes(classification)
    native = tiny_model(classification)
    processor = SimpleNamespace(
        size={"shortest_edge": 256}, crop_size=crop_type(height=224, width=224), do_center_crop=center_crop
    )
    config = cls._update_rbln_config([processor], native, native.config, config_cls(image_size=explicit_size))
    expected = explicit_size or (224 if center_crop else 256)
    height, width = (expected, expected) if isinstance(expected, int) else expected
    assert config.compile_cfgs[0].input_info[0][1] == [1, 3, height, width]


@pytest.mark.parametrize("classification", [False, True])
def test_auto_discovery(classification, monkeypatch):
    from optimum.rbln import RBLNAutoModel, RBLNAutoModelForImageClassification

    cls, _ = rbln_classes(classification)
    monkeypatch.setattr(cls, "from_model", classmethod(lambda cls, model, **kwargs: cls))
    auto = RBLNAutoModelForImageClassification if classification else RBLNAutoModel
    assert auto.from_model(tiny_model(classification)) is cls


@pytest.mark.parametrize("classification", [False, True])
def test_native_processor_pixels_match_default_compile_shape(classification):
    from PIL import Image
    from transformers import BitImageProcessor

    cls, config_cls = rbln_classes(classification)
    native = tiny_model(classification)
    processor = BitImageProcessor(size={"shortest_edge": 256}, crop_size={"height": 224, "width": 224})
    pixels = processor(images=Image.new("RGB", (480, 320)), return_tensors="pt").pixel_values
    config = cls._update_rbln_config([processor], native, native.config, config_cls())
    assert config.compile_cfgs[0].input_info[0][1] == list(pixels.shape)


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("size", [(28, 28), (28, 42), (42, 56), {"height": 28, "width": 42}])
def test_position_encoding_is_precomputed_once(classification, size, monkeypatch):
    from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings

    torch.manual_seed(42)
    native = tiny_model(classification).float().eval()
    height, width = (size["height"], size["width"]) if isinstance(size, dict) else size
    pixels = torch.randn(2, 3, height, width)
    with torch.no_grad():
        expected = native(pixels).to_tuple()
    interpolate = Dinov2Embeddings.interpolate_pos_encoding
    calls = []

    def record(self, embeddings, height, width):
        calls.append((embeddings.dtype, height, width))
        return interpolate(self, embeddings, height, width)

    monkeypatch.setattr(Dinov2Embeddings, "interpolate_pos_encoding", record)
    cls, config_cls = rbln_classes(classification)
    config = cls._update_rbln_config(None, native, native.config, config_cls(image_size=size, batch_size=2))
    wrapper = cls._wrap_model_if_needed(native, config)
    assert calls == [(torch.float32, height, width)]
    with torch.no_grad():
        for _ in range(2):
            for a, e in zip(wrapper(pixels), expected, strict=True):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
    assert calls == [(torch.float32, height, width)]
    exported = torch.export.export(wrapper, (pixels,))
    assert not any("upsample_bicubic2d" in str(node.target) for node in exported.graph.nodes)
