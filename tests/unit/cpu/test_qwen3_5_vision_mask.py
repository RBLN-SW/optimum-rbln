from types import SimpleNamespace

import torch

from optimum.rbln.configuration_utils import RBLNCompileConfig
from optimum.rbln.transformers.models.qwen3_5.configuration_qwen3_5 import RBLNQwen3_5VisionModelConfig
from optimum.rbln.transformers.models.qwen3_5.modeling_qwen3_5 import RBLNQwen3_5VisionModel


def _to_bias(mask: torch.Tensor) -> torch.Tensor:
    """What the vision wrapper does with the 0/1 mask before attention."""
    return (1.0 - mask) * torch.finfo(mask.dtype).min


def _model_declaring(*mask_shapes: list[int]) -> RBLNQwen3_5VisionModel:
    """A vision model carrying only the compiled input_info the mask path reads."""
    input_infos = [
        [("hidden_states", [shape[-1], 16], torch.float32), ("attn_mask", shape, torch.float32)]
        for shape in mask_shapes
    ]
    model = RBLNQwen3_5VisionModel.__new__(RBLNQwen3_5VisionModel)
    model.rbln_config = SimpleNamespace(compile_cfgs=[RBLNCompileConfig(input_info=input_infos)])
    return model


def test_key_padding_mask_matches_square_mask_on_valid_rows():
    torch.manual_seed(0)
    seq_len, bucket, dim = 5, 8, 16
    hidden = torch.randn(seq_len, dim)
    rotary = (torch.randn(seq_len, 4), torch.randn(seq_len, 4))

    # The [S, S] path is deprecated: from optimum-rbln 0.13.0 it raises, and this call is the
    # reminder to delete it with _mask_padded_keys_and_queries and _declared_mask_rows.
    _, _, key_mask, valid_len = _model_declaring([1, 1, 1, bucket])._pad_hidden_states(hidden, rotary, bucket)
    _, _, square_mask, _ = _model_declaring([1, 1, bucket, bucket])._pad_hidden_states(hidden, rotary, bucket)
    assert key_mask.shape == (1, 1, 1, bucket)
    assert square_mask.shape == (1, 1, bucket, bucket)

    # The padded query rows differ by construction, but their outputs are dropped; the valid rows must match.
    q, k, v = (torch.randn(1, 2, bucket, dim) for _ in range(3))
    by_key = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=_to_bias(key_mask))
    by_square = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=_to_bias(square_mask))
    torch.testing.assert_close(by_key[:, :, :valid_len], by_square[:, :, :valid_len])


def test_declared_mask_rows_matches_the_bucket_by_size():
    # A bucket cannot be found by position: RBLNQwen3_5VisionModelConfig sorts max_seq_len descending
    # while forward() searches it ascending.
    model = _model_declaring([1, 1, 16, 16], [1, 1, 1, 8])

    assert model._declared_mask_rows(16) == 16
    assert model._declared_mask_rows(8) == 1


def test_a_fresh_compile_declares_and_reads_back_one_row():
    """What _update_rbln_config compiles and what _pad_hidden_states feeds must agree."""
    buckets = [5120, 16384]
    rbln_config = RBLNQwen3_5VisionModelConfig(max_seq_len=buckets, batch_size=1)
    RBLNQwen3_5VisionModel._update_rbln_config(
        model_config=SimpleNamespace(hidden_size=1152, num_heads=16), rbln_config=rbln_config
    )

    model = RBLNQwen3_5VisionModel.__new__(RBLNQwen3_5VisionModel)
    model.rbln_config = rbln_config
    for bucket in buckets:
        _, _, mask, _ = model._pad_hidden_states(torch.randn(8, 1152), (torch.randn(8, 4), torch.randn(8, 4)), bucket)
        assert mask.shape == (1, 1, 1, bucket)
