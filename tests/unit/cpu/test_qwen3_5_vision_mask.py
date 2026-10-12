from types import SimpleNamespace

import torch

from optimum.rbln.transformers.models.qwen3_5.configuration_qwen3_5 import RBLNQwen3_5VisionModelConfig
from optimum.rbln.transformers.models.qwen3_5.modeling_qwen3_5 import RBLNQwen3_5VisionModel


def _to_bias(mask: torch.Tensor) -> torch.Tensor:
    """What the vision wrapper does with the 0/1 mask before attention."""
    return (1.0 - mask) * torch.finfo(mask.dtype).min


def _model_declaring(*mask_shapes: list[int]) -> RBLNQwen3_5VisionModel:
    """A real instance carrying only the compiled input_info the mask path reads, left uninitialized
    so the test does not need an artifact."""
    input_infos = [
        [("hidden_states", [shape[-1], 16], torch.float32), ("attn_mask", shape, torch.float32)]
        for shape in mask_shapes
    ]
    model = RBLNQwen3_5VisionModel.__new__(RBLNQwen3_5VisionModel)
    model.rbln_config = SimpleNamespace(compile_cfgs=[SimpleNamespace(input_info=input_infos)])
    return model


def test_key_padding_mask_matches_square_mask_on_valid_rows():
    torch.manual_seed(0)
    seq_len, bucket, dim = 5, 8, 16
    hidden = torch.randn(seq_len, dim)
    rotary = (torch.randn(seq_len, 4), torch.randn(seq_len, 4))

    # The mask each export declares, built the way forward() builds it.
    this_export = _model_declaring([1, 1, 1, bucket])
    older_export = _model_declaring([1, 1, bucket, bucket])
    _, _, key_mask, valid_len = this_export._pad_hidden_states(hidden, rotary, bucket)
    _, _, square_mask, _ = older_export._pad_hidden_states(hidden, rotary, bucket)

    assert key_mask.shape == (1, 1, 1, bucket)
    assert square_mask.shape == (1, 1, bucket, bucket)
    assert key_mask[0, 0, 0].tolist() == [1.0] * seq_len + [0.0] * (bucket - seq_len)

    # The padded query rows differ by construction, but their outputs are dropped; the valid rows must match.
    q, k, v = (torch.randn(1, 2, bucket, dim) for _ in range(3))
    by_key = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=_to_bias(key_mask))
    by_square = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=_to_bias(square_mask))
    torch.testing.assert_close(by_key[:, :, :valid_len], by_square[:, :, :valid_len])


def test_a_full_bucket_is_unmasked():
    hidden = torch.randn(8, 16)
    rotary = (torch.randn(8, 4), torch.randn(8, 4))

    model = _model_declaring([1, 1, 1, 8])
    _, _, mask, valid_len = model._pad_hidden_states(hidden, rotary, 8)

    assert valid_len == 8
    assert (mask == 1.0).all()


def test_declared_mask_rows_follows_the_export():
    # This export declares the key-padding mask; one made before it declares max_seq_len rows.
    assert _model_declaring([1, 1, 1, 8])._declared_mask_rows(8) == 1
    assert _model_declaring([1, 1, 8, 8])._declared_mask_rows(8) == 8


def test_declared_mask_rows_matches_the_bucket_by_size():
    # RBLNQwen3_5VisionModelConfig sorts max_seq_len descending while forward() searches it ascending,
    # so the bucket must be found by its size rather than by position.
    model = _model_declaring([1, 1, 16, 16], [1, 1, 1, 8])

    assert model._declared_mask_rows(16) == 16
    assert model._declared_mask_rows(8) == 1


def test_declared_mask_rows_accepts_a_single_bucket_input_info():
    # RBLNCompileConfig keeps one bucket as a bare input_info rather than a list of them.
    single = RBLNQwen3_5VisionModel.__new__(RBLNQwen3_5VisionModel)
    single.rbln_config = SimpleNamespace(
        compile_cfgs=[
            SimpleNamespace(
                input_info=[("hidden_states", [8, 16], torch.float32), ("attn_mask", [1, 1, 1, 8], torch.float32)]
            )
        ]
    )

    assert single._declared_mask_rows(8) == 1


def test_a_fresh_export_declares_and_reads_back_one_row():
    """What _update_rbln_config compiles and what _pad_hidden_states feeds must agree."""
    buckets = [5120, 16384]
    rbln_config = RBLNQwen3_5VisionModelConfig(max_seq_len=buckets, batch_size=1)
    model_config = SimpleNamespace(hidden_size=1152, num_heads=16)

    RBLNQwen3_5VisionModel._update_rbln_config(model_config=model_config, rbln_config=rbln_config)

    model = RBLNQwen3_5VisionModel.__new__(RBLNQwen3_5VisionModel)
    model.rbln_config = rbln_config
    for bucket in buckets:
        assert model._declared_mask_rows(bucket) == 1
        _, _, mask, _ = model._pad_hidden_states(torch.randn(8, 1152), (torch.randn(8, 4), torch.randn(8, 4)), bucket)
        assert mask.shape == (1, 1, 1, bucket)
