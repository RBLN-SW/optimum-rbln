import pytest
import torch
from transformers import MPNetConfig, MPNetModel

from optimum.rbln import RBLNMPNetModel, RBLNMPNetModelConfig


@pytest.mark.parametrize("add_pooling_layer", [True, False])
def test_wrapper_matches_hf(add_pooling_layer):
    torch.manual_seed(42)
    config = MPNetConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64
    )
    model = MPNetModel(config, add_pooling_layer=add_pooling_layer).eval()
    input_ids = torch.randint(4, 64, (2, 16))
    attention_mask = (torch.arange(16) < torch.tensor([[7], [16]])).long()

    with torch.no_grad():
        expected = [t for t in model(input_ids, attention_mask, return_dict=False) if t is not None]
        wrapper = RBLNMPNetModel._wrap_model_if_needed(model, RBLNMPNetModelConfig(max_seq_len=16))
        actual = wrapper(input_ids, attention_mask)

    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e)
