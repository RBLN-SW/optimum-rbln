from types import SimpleNamespace

import pytest
import torch
from transformers import CLIPVisionConfig, CLIPVisionModel

from optimum.rbln import RBLNCLIPVisionModel, RBLNCLIPVisionModelConfig
from optimum.rbln.transformers.models.clip.modeling_clip import _VisionEncoder
from optimum.rbln.transformers.models.llava_next.modeling_llava_next import LoopVisionTower


def tiny_config():
    return CLIPVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2, image_size=8, patch_size=4
    )


@pytest.mark.parametrize("indices", [[-2], [0, -1], [2, 1, 2]])
def test_selected_encoder_returns_only_requested_features(indices):
    torch.manual_seed(7)
    model = CLIPVisionModel(tiny_config()).eval()
    pixels = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        expected = model(pixels, output_hidden_states=True).hidden_states
        output = _VisionEncoder(model, False, True, False, output_hidden_state_indices=indices)(pixels)
    assert len(output) == len(indices)
    for tensor, index in zip(output, indices, strict=True):
        torch.testing.assert_close(tensor, expected[index], rtol=0, atol=0)


@pytest.mark.parametrize("indices", [[], [True], ["1"], [-5], [4]])
def test_invalid_feature_indices_are_rejected(indices):
    with pytest.raises(ValueError, match="output_hidden_state_indices"):
        cfg = RBLNCLIPVisionModelConfig(output_hidden_states=True, output_hidden_state_indices=indices)
        RBLNCLIPVisionModel._update_rbln_config([], model_config=tiny_config(), rbln_config=cfg)


def test_selected_output_does_not_require_last_or_pooler_ports():
    cfg = RBLNCLIPVisionModelConfig(output_hidden_states=True, output_hidden_state_indices=[-2])
    cfg = RBLNCLIPVisionModel._update_rbln_config([], model_config=tiny_config(), rbln_config=cfg)
    feature = torch.randn(1, 5, 16)
    stub = SimpleNamespace(rbln_config=cfg, config=tiny_config())
    output = RBLNCLIPVisionModel._prepare_output(stub, [feature], True)
    assert output.last_hidden_state is None and output.pooler_output is None
    assert torch.equal(output.hidden_states[0], feature)


def test_loop_selected_outputs_preserve_crop_order():
    class Runtime:
        def __call__(self, pixels, out):
            out[0].copy_(pixels.flatten(1)[:, :1, None].expand_as(out[0]))
            return out[0]

    cfg = RBLNCLIPVisionModelConfig(output_hidden_states=True, output_hidden_state_indices=[2])
    tower = LoopVisionTower(SimpleNamespace(model=[Runtime()], rbln_config=cfg))
    buffer = torch.empty(3, 5, 16)
    pixels = torch.arange(3.0).reshape(3, 1, 1, 1)
    output = tower(pixels, out=[buffer])
    assert output.last_hidden_state is None and output.pooler_output is None
    torch.testing.assert_close(output.hidden_states[0][:, 0, 0], torch.tensor([0.0, 1.0, 2.0]))


def test_default_encoder_keeps_full_output_contract():
    model = CLIPVisionModel(tiny_config()).eval()
    with torch.no_grad():
        output = _VisionEncoder(model, False, True, False)(torch.randn(1, 3, 8, 8))
    assert output[0].shape == (1, 5, 16)
    assert output[1].shape == (1, 16)
    assert len(output[2]) == 4


@pytest.mark.parametrize("indices", [None, [2, 1]])
@pytest.mark.parametrize("flat_pixels", [False, True])
def test_llava_selects_requested_layers_in_order_for_multiple_crops(indices, flat_pixels):
    from optimum.rbln.transformers.models.llava_next.modeling_llava_next import (
        RBLNLlavaNextForConditionalGeneration,
    )

    class VisionRuntime:
        def __call__(self, pixels, out):
            if indices is None:
                for i, tensor in enumerate(out[2:]):
                    tensor.fill_(i)
            else:
                for i, tensor in zip(indices, out, strict=True):
                    tensor.fill_(i)
            return out

    cfg = RBLNCLIPVisionModelConfig(output_hidden_states=True, output_hidden_state_indices=indices)
    config = SimpleNamespace(
        vision_config=tiny_config(), text_config=SimpleNamespace(hidden_size=8), image_grid_pinpoints=[[8, 8]]
    )

    def projector(features, out):
        out[0].copy_(features[:, :, :8] + 10 * features[:, :, 16:24])
        return out[0]

    model = SimpleNamespace(
        config=config,
        rbln_config=SimpleNamespace(vision_tower=cfg, dtype=torch.float32),
        vision_tower=LoopVisionTower(SimpleNamespace(model=[VisionRuntime()], rbln_config=cfg)),
        multi_modal_projector=projector,
    )
    features = RBLNLlavaNextForConditionalGeneration.get_image_features(
        model,
        torch.zeros((2, 3, 8, 8) if flat_pixels else (1, 2, 3, 8, 8)),
        torch.tensor([[8, 8]]),
        [-3, -2],
        "default",
    )
    assert len(features) == 1
    torch.testing.assert_close(features[0], torch.full((2, 4, 8), 21.0))


def test_llava_rejects_uncompiled_layer_before_runtime_call():
    from optimum.rbln.transformers.models.llava_next.modeling_llava_next import (
        RBLNLlavaNextForConditionalGeneration,
    )

    cfg = RBLNCLIPVisionModelConfig(output_hidden_states=True, output_hidden_state_indices=[2])
    model = SimpleNamespace(
        config=SimpleNamespace(vision_config=tiny_config(), image_grid_pinpoints=[[8, 8]]),
        rbln_config=SimpleNamespace(vision_tower=cfg),
    )
    with pytest.raises(ValueError, match="was not compiled"):
        RBLNLlavaNextForConditionalGeneration.get_image_features(
            model, torch.zeros(1, 2, 3, 8, 8), torch.tensor([[8, 8]]), 1, "default"
        )
