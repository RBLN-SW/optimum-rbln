import copy

import pytest
import torch
from diffusers import Flux2Transformer2DModel

from optimum.rbln.diffusers.configurations.models.configuration_transformer_flux2 import (
    RBLNFlux2Transformer2DModelConfig,
)
from optimum.rbln.diffusers.models.transformers.transformer_flux2 import (
    Flux2TransformerWrapper,
    RBLNFlux2Transformer2DModel,
    pack_rotary_embedding,
)


@pytest.mark.parametrize(("batch", "image_tokens"), [(1, 6), (2, 12), (1, 18)])
def test_transformer_wrapper_preserves_native_forward_and_trained_state(batch, image_tokens):
    torch.manual_seed(19)
    native = Flux2Transformer2DModel(
        in_channels=8,
        out_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=24,
        timestep_guidance_channels=16,
        mlp_ratio=2,
        axes_dims_rope=(4, 4, 4, 4),
        guidance_embeds=False,
    ).eval()
    before = copy.deepcopy(native.state_dict())
    original_position_module = native.pos_embed
    image_ids = torch.zeros(batch, image_tokens, 4)
    image_ids[:, :, 0] = torch.arange(image_tokens) // 6
    image_ids[:, :, 2] = torch.arange(image_tokens)
    text_ids = torch.zeros(batch, 3, 4)
    text_ids[:, :, 3] = torch.arange(3)
    hidden = torch.randn(batch, image_tokens, 8)
    context = torch.randn(batch, 3, 24)
    timestep = torch.linspace(0.1, 0.8, batch)
    wrapper = Flux2TransformerWrapper(native).eval()
    with torch.no_grad():
        expected = native(hidden, context, timestep, image_ids, text_ids, return_dict=False)[0]
        actual = wrapper(
            hidden,
            context,
            timestep,
            pack_rotary_embedding(native.pos_embed, image_ids),
            pack_rotary_embedding(native.pos_embed, text_ids),
        )[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert native.pos_embed is original_position_module
    assert wrapper.model.pos_embed is not original_position_module
    for name, value in native.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        torch.testing.assert_close(wrapper.model.state_dict()[name], before[name], rtol=0, atol=0)


def test_rotary_packing_preserves_native_values():
    from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed

    position = Flux2PosEmbed(theta=2000, axes_dim=[4, 4, 4, 4])
    ids = torch.tensor([[0, 0, 4095, 1], [10, 2, 17, 511]], dtype=torch.float32)
    expected_cos, expected_sin = position(ids)
    packed = pack_rotary_embedding(position, ids.unsqueeze(0))
    cosine, sine = packed.chunk(2, dim=-1)
    torch.testing.assert_close(cosine, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(sine, expected_sin, rtol=0, atol=0)


@pytest.mark.parametrize("references", [0, 1, 2])
def test_native_profile_preserves_full_image_and_text_capacity(references):
    from types import SimpleNamespace

    cfg = RBLNFlux2Transformer2DModelConfig(reference_image_count=references, batch_size=2, npu="RBLN-CR13")
    native_config = SimpleNamespace(in_channels=128, joint_attention_dim=7680, attention_head_dim=128)
    actual = RBLNFlux2Transformer2DModel._update_rbln_config([], None, native_config, cfg)
    profile = {name: (shape, dtype) for name, shape, dtype in actual.compile_cfgs[0].input_info}
    assert profile["hidden_states"][0] == [2, 4096 * (1 + references), 128]
    assert profile["encoder_hidden_states"][0] == [2, 512, 7680]
    assert profile["image_rotary_emb"][0] == [4096 * (1 + references), 256]
    assert profile["text_rotary_emb"][0] == [512, 256]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": True}, "batch_size"),
        ({"reference_image_count": -1}, "reference_image_count"),
        ({"image_size": (1023, 1024)}, "multiples of 16"),
        ({"reference_image_size": (0, 1024)}, "multiples of 16"),
        ({"prompt_embed_length": 513}, "prompt_embed_length"),
    ],
)
def test_invalid_static_profiles_fail_before_export(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RBLNFlux2Transformer2DModelConfig(**kwargs)
