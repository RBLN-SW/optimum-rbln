import torch
from transformers import ModernBertConfig, ModernBertModel

from optimum.rbln import RBLNModernBertModel, RBLNModernBertModelConfig


def test_wrapper_matches_hf():
    torch.manual_seed(42)
    # Layer 0 uses global attention and layer 1 sliding-window attention (global_attn_every_n_layers=3).
    config = ModernBertConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        local_attention=8,
        global_attn_every_n_layers=3,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
        reference_compile=False,
    )
    config._attn_implementation = "sdpa"
    model = ModernBertModel(config).eval()
    input_ids = torch.randint(1, 128, (2, 24))
    attention_mask = (torch.arange(24) < torch.tensor([[17], [24]])).long()

    with torch.no_grad():
        expected = model(input_ids, attention_mask, return_dict=False)
        wrapper = RBLNModernBertModel._wrap_model_if_needed(model, RBLNModernBertModelConfig(max_seq_len=24))
        actual = wrapper(input_ids, attention_mask)

    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e)
