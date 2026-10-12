import pytest
import torch
from transformers import SwinConfig, SwinForImageClassification


@pytest.mark.parametrize("size", [(32, 32), (32, 64)])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("return_dict", [False, True])
def test_swin_classifier_preserves_shifted_window_logits(size, batch, return_dict):
    import optimum.rbln as rbln

    torch.manual_seed(42)
    config = SwinConfig(image_size=size, embed_dim=16, depths=[2, 2], num_heads=[2, 4], window_size=4, num_labels=5)
    reference = SwinForImageClassification(config).eval()
    pixels = torch.randn(batch, 3, *size)
    with torch.no_grad():
        expected = reference(pixels).logits
    cls = rbln.RBLNSwinForImageClassification
    cfg = rbln.RBLNSwinForImageClassificationConfig(image_size=size, batch_size=batch)
    cfg = cls._update_rbln_config(None, reference, config, cfg)
    wrapper = cls._wrap_model_if_needed(reference, cfg)
    runtime = object.__new__(cls)
    runtime.config = config
    runtime.rbln_config = cfg
    runtime.model = [wrapper]
    with torch.no_grad():
        actual = runtime(pixel_values=pixels, return_dict=return_dict)
    actual = actual.logits if return_dict else actual[0]
    torch.testing.assert_close(actual, expected)
    assert cfg.compile_cfgs[0].input_info == [("pixel_values", [batch, 3, *size], "float32")]


def test_swin_classifier_auto_selects_export(monkeypatch):
    import optimum.rbln as rbln

    cls = rbln.RBLNSwinForImageClassification
    monkeypatch.setattr(cls, "from_model", classmethod(lambda cls, model, **kwargs: cls))
    reference = SwinForImageClassification(SwinConfig(embed_dim=16, depths=[1, 1], num_heads=[2, 4]))
    assert rbln.RBLNAutoModelForImageClassification.from_model(reference) is cls


@pytest.mark.parametrize("size", [(32, 32), (28, 44), (32, 48)])
def test_swin_classifier_mask_is_precomputed_before_tracing(size):
    import optimum.rbln as rbln

    torch.manual_seed(42)
    config = SwinConfig(image_size=size, embed_dim=16, depths=[2, 2], num_heads=[2, 4], window_size=4, num_labels=5)
    model = SwinForImageClassification(config).eval()
    pixels = torch.randn(1, 3, *size)
    with torch.no_grad():
        expected = model(pixels).logits
    cls = rbln.RBLNSwinForImageClassification
    cfg = rbln.RBLNSwinForImageClassificationConfig(image_size=size)
    cfg = cls._update_rbln_config(None, model, config, cfg)
    wrapper = cls._wrap_model_if_needed(model, cfg)
    for stage in wrapper.encoder.layers:
        for block in stage.blocks:
            assert "rbln_attn_mask" in block._buffers
            mask = block.rbln_attn_mask
            if mask is not None:
                assert block.get_attn_mask(0, 0, mask.dtype, mask.device) is mask
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, pixels)
        torch.testing.assert_close(traced(pixels), expected)
    assert "aten::copy_" not in str(traced.inlined_graph)


@pytest.mark.parametrize("enabled", [False, True])
def test_swin_classifier_annotation_export_option(monkeypatch, enabled):
    import rebel

    import optimum.rbln as rbln

    cls = rbln.RBLNSwinForImageClassification
    model = SwinForImageClassification(SwinConfig(embed_dim=16, depths=[2, 2], num_heads=[2, 4], window_size=4)).eval()
    cfg = rbln.RBLNSwinForImageClassificationConfig(
        image_size=32, batch_size=8, create_runtimes=False, use_rsd_annotation=enabled
    )
    cfg = cls._update_rbln_config(None, model, model.config, cfg)
    captured = {}
    marker = object()

    def compile_model(model, **kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(rebel, "compile_from_torch", compile_model)
    assert cls.get_compiled_model(model, cfg) is marker
    assert captured["input_info"] == [("pixel_values", [8, 3, 32, 32], "float32")]
    if enabled:
        assert isinstance(captured["compile_context"], rebel.CompileContext)
        assert captured["use_weight_sharing"] is True
    else:
        assert "compile_context" not in captured
        assert "use_weight_sharing" not in captured


@pytest.mark.parametrize("invalid", [None, 1, "true"])
def test_swin_classifier_annotation_option_requires_bool(invalid):
    from optimum.rbln import RBLNSwinForImageClassificationConfig

    with pytest.raises(ValueError, match="use_rsd_annotation.*bool"):
        RBLNSwinForImageClassificationConfig(use_rsd_annotation=invalid)


def test_swin_classifier_annotation_option_roundtrip(tmp_path):
    from optimum.rbln import RBLNSwinForImageClassificationConfig

    cfg = RBLNSwinForImageClassificationConfig(use_rsd_annotation=True, image_size=224, batch_size=8)
    cfg.save(tmp_path)
    restored = RBLNSwinForImageClassificationConfig.from_pretrained(tmp_path)
    assert restored.use_rsd_annotation is True
    assert restored.batch_size == 8
    assert RBLNSwinForImageClassificationConfig().use_rsd_annotation is False
