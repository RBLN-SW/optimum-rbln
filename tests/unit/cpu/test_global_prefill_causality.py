from types import SimpleNamespace

import pytest
import torch

from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import AttentionOp, FlashAttentionOp


@pytest.mark.parametrize("op_type", [AttentionOp, FlashAttentionOp])
@pytest.mark.parametrize("phase", ["prefill", "image_prefill", "decode"])
@pytest.mark.parametrize("image_prefill", [False, True])
@pytest.mark.parametrize("position_ids", [False, True])
@pytest.mark.parametrize("attention_mask", [False, True])
def test_global_prefill_causality(monkeypatch, op_type, phase, image_prefill, position_ids, attention_mask):
    op = op_type.__new__(op_type)
    torch.nn.Module.__init__(op)
    op.phase = phase
    op.quantization = None
    op.use_attention_mask = attention_mask
    op.num_heads, op.num_key_value_heads, op.head_dim = 4, 1, 256
    op.kvcache_partition_size = 512
    op.rbln_config = SimpleNamespace(use_image_prefill=image_prefill, use_position_ids=position_ids)
    length = 1 if phase == "decode" else 16
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return torch.zeros(1, 1, 4, length, 256)

    monkeypatch.setattr(torch.ops.rbln_custom_ops, op.get_attn_op_name(), capture)
    op(
        torch.zeros(1, 4, length, 256),
        torch.zeros(1, 1, length, 256),
        torch.zeros(1, 1, length, 256),
        torch.ones(1, 1, length, 512),
        torch.zeros(1, 1, 512, 256),
        torch.zeros(1, 1, 512, 256),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.tensor(0.0625),
        torch.tensor([0], dtype=torch.int16),
        512,
    )
    if phase == "decode" or (not image_prefill and attention_mask and not position_ids):
        assert "is_bidirectional" not in seen
    else:
        assert seen["is_bidirectional"] is (
            phase == "image_prefill" and (image_prefill or (attention_mask and position_ids))
        )
