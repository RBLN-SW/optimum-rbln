from types import SimpleNamespace

import pytest
import torch

from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import SlidingWindowAttentionOp


@pytest.mark.parametrize("phase", ["prefill", "image_prefill", "decode"])
@pytest.mark.parametrize("image_prefill", [False, True])
@pytest.mark.parametrize("position_ids", [False, True])
def test_sliding_prefill_causality(monkeypatch, phase, image_prefill, position_ids):
    # Capture the SDK boundary without compiling a graph or allocating an NPU.
    op = SlidingWindowAttentionOp.__new__(SlidingWindowAttentionOp)
    torch.nn.Module.__init__(op)
    op.phase = phase
    op.quantization = None
    op.use_attention_mask = True
    op.num_heads, op.num_key_value_heads, op.head_dim = 4, 1, 256
    op.rbln_config = SimpleNamespace(use_image_prefill=image_prefill, use_position_ids=position_ids)
    seen = {}
    length = 1 if phase == "decode" else 16

    def capture(**kwargs):
        seen.update(kwargs)
        return torch.zeros(1, 1, 4, length, 256)

    monkeypatch.setattr(torch.ops.rbln_custom_ops, op.get_attn_op_name(), capture)
    op(
        torch.zeros(1, 4, length, 256),
        torch.zeros(1, 1, length, 256),
        torch.zeros(1, 1, length, 256),
        None,
        torch.zeros(1, 1, 512, 256),
        torch.zeros(1, 1, 512, 256),
        (torch.zeros(1, 1, dtype=torch.int32), torch.tensor([[length]], dtype=torch.int32)),
        torch.tensor(0.0625),
        torch.tensor([0], dtype=torch.int16),
        512,
    )
    if phase == "decode":
        assert "is_bidirectional" not in seen
    else:
        assert seen["is_bidirectional"] is (phase == "image_prefill" and (image_prefill or position_ids))
