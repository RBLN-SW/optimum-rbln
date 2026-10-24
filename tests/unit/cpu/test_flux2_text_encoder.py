import pytest
import torch
from diffusers import Flux2KleinPipeline
from transformers import Qwen3Config, Qwen3ForCausalLM

from optimum.rbln.diffusers.models.transformers.flux2_text_encoder import (
    Flux2KleinTextEncoderWrapper,
    RBLNFlux2KleinTextEncoder,
)


@pytest.mark.parametrize("masked", [False, True])
def test_text_wrapper_preserves_every_native_hidden_state(masked):
    torch.manual_seed(31)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
    )
    native = Qwen3ForCausalLM(config).eval()
    ids = torch.randint(1, 64, (2, 17))
    mask = torch.ones_like(ids)
    if masked:
        mask[0, -3:] = 0
        mask[1, :2] = 0
    before = {name: value.clone() for name, value in native.state_dict().items()}
    with torch.inference_mode():
        expected = native(ids, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
        actual = Flux2KleinTextEncoderWrapper(native)(ids, mask)
    assert len(actual) == len(expected) == config.num_hidden_layers + 1
    for value, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, target, rtol=0, atol=0)
    for name, value in native.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_native_prompt_encoding_preserves_default_layers_and_mixed_padding():
    torch.manual_seed(37)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=28,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=512,
    )
    native = Qwen3ForCausalLM(config).eval()

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, text, max_length, **kwargs):
            length = min(len(text), max_length)
            ids = torch.zeros(1, max_length, dtype=torch.long)
            ids[0, :length] = torch.arange(length) % 63 + 1
            return {"input_ids": ids, "attention_mask": (ids != 0).long()}

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.ModuleList([Flux2KleinTextEncoderWrapper(native)])
            self.config = config

        @property
        def dtype(self):
            return native.dtype

        @property
        def device(self):
            return native.device

        forward = RBLNFlux2KleinTextEncoder.forward

    kwargs = {"tokenizer": Tokenizer(), "prompt": ["short", "a longer padded prompt"], "max_sequence_length": 32}
    with torch.inference_mode():
        expected = Flux2KleinPipeline._get_qwen3_prompt_embeds(native, **kwargs)
        actual = Flux2KleinPipeline._get_qwen3_prompt_embeds(Adapter(), **kwargs)
    assert actual.shape == (2, 32, 48)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
