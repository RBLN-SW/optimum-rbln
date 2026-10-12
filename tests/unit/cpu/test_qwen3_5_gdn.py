from types import SimpleNamespace

import pytest
import torch
from rebel.ops.torch_custom_ops.gdn import unblock_state
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet as HFGDN

from optimum.rbln.transformers.models.qwen3_5.modeling_qwen3_5 import _qwen3_5_linear_state_shapes
from optimum.rbln.transformers.models.qwen3_5.qwen3_5_architecture import Qwen3_5GatedDeltaNet


def config(ratio):
    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2 * ratio,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )


def test_state_layout_and_fallback():
    cfg = config(2)
    opts = SimpleNamespace(prefill_chunk_size=128, gdn_chunk_size=128)
    assert _qwen3_5_linear_state_shapes(cfg, 3, opts)[1] == (3, 2, 2, 2, 128, 64)
    opts.gdn_chunk_size = 64
    assert _qwen3_5_linear_state_shapes(cfg, 3, opts)[1] == (3, 2, 2, 2, 128, 64)
    cfg.linear_key_head_dim = 64
    assert _qwen3_5_linear_state_shapes(cfg, 3, opts)[1] == (3, 256, 128)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("chunk", [64, 128])
def test_slot_reset_and_prefill_decode_handoff(ratio, chunk):
    torch.manual_seed(42)
    torch.set_num_threads(4)
    cfg = config(ratio)
    opts = SimpleNamespace(prefill_chunk_size=128, gdn_chunk_size=chunk)
    native = HFGDN(cfg, 0).eval()
    actual = Qwen3_5GatedDeltaNet(native, opts, 0).eval()
    reference = Qwen3_5GatedDeltaNet(native, opts, 0).eval()
    reference.use_gdn_custom_kernel = False
    conv = torch.zeros(1, 3, native.conv_dim)
    old_state = torch.zeros(1, 2 * ratio * 128, 128)
    cache = torch.randn(3, 2, ratio, 2, 128, 64)
    untouched = cache[[0, 2]].clone()
    slot = torch.tensor(1, dtype=torch.int16)
    with torch.no_grad():
        for step, seq in enumerate([128, 128, 1, 1]):
            actual.phase = reference.phase = "prefill" if seq > 1 else "decode"
            x = torch.randn(1, seq, 64) * 0.1
            extras = (
                {
                    "query_position": torch.tensor(seq - 1, dtype=torch.int16),
                    "valid_mask": torch.ones(1, seq, 1),
                    "conv_state_mask": torch.ones_like(conv),
                }
                if seq > 1
                else {}
            )
            expected, new_conv, old_state = reference(x, conv, old_state, **extras)
            if seq > 1:
                extras["recurrent_state_mask"] = torch.tensor([0.0 if step == 0 else 1.0])
            result, new_conv_actual, returned_state = actual(x, conv, cache, state_slot=slot, **extras)
            if chunk == 64 and seq > 1:
                torch.testing.assert_close(
                    unblock_state(returned_state).reshape_as(old_state), old_state, rtol=2e-5, atol=2e-6
                )
                cache[1:2].copy_(returned_state)
            torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(new_conv_actual, new_conv)
            torch.testing.assert_close(
                unblock_state(cache[1:2]).reshape_as(old_state), old_state, rtol=2e-5, atol=2e-6
            )
            torch.testing.assert_close(cache[[0, 2]], untouched, rtol=0, atol=0)
            conv = new_conv
