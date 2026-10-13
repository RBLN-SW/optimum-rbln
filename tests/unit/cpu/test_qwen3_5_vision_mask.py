from types import SimpleNamespace

import torch

from optimum.rbln.transformers.models.qwen3_5.modeling_qwen3_5 import RBLNQwen3_5VisionModel


def _to_bias(mask: torch.Tensor) -> torch.Tensor:
    return (1.0 - mask) * torch.finfo(mask.dtype).min


def test_key_padding_mask_matches_square_mask_on_valid_rows():
    torch.manual_seed(0)
    seq_len, bucket, dim = 5, 8, 16
    hidden = torch.randn(seq_len, dim)
    rotary = (torch.randn(seq_len, 4), torch.randn(seq_len, 4))

    _, _, key_mask, valid_len = RBLNQwen3_5VisionModel._pad_hidden_states(hidden, rotary, bucket)
    _, _, square_mask, _ = RBLNQwen3_5VisionModel._pad_hidden_states(hidden, rotary, bucket, square_mask=True)

    assert key_mask.shape == (1, 1, 1, bucket)
    assert square_mask.shape == (1, 1, bucket, bucket)
    assert key_mask[0, 0, 0].tolist() == [1.0] * seq_len + [0.0] * (bucket - seq_len)

    q, k, v = (torch.randn(1, 2, bucket, dim) for _ in range(3))
    by_key = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=_to_bias(key_mask))
    by_square = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=_to_bias(square_mask))
    torch.testing.assert_close(by_key[:, :, :valid_len], by_square[:, :, :valid_len])


def test_older_exports_keep_the_square_mask():
    def model_with(mask_shape):
        input_info = [[("hidden_states", [8, 16], torch.float32), ("attn_mask", mask_shape, torch.float32)]]
        return SimpleNamespace(rbln_config=SimpleNamespace(compile_cfgs=[SimpleNamespace(input_info=input_info)]))

    assert RBLNQwen3_5VisionModel._takes_square_mask(model_with([1, 1, 8, 8]), 0)
    assert not RBLNQwen3_5VisionModel._takes_square_mask(model_with([1, 1, 1, 8]), 0)
