import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, Gemma4ForCausalLM, Gemma4TextConfig

from optimum.rbln import RBLNGemma3ForCausalLMConfig, RBLNGemma4ForCausalLMConfig
from optimum.rbln.transformers.models.gemma3.gemma3_architecture import Gemma3Attention
from optimum.rbln.transformers.models.gemma4.gemma4_architecture import Gemma4TextAttention


def _capture_is_bidirectional(monkeypatch, op, phase):
    captured = {}

    def fake_op(**kwargs):
        captured.update(kwargs)
        return torch.zeros(1, op.num_heads, 1, op.head_dim)

    monkeypatch.setattr(torch.ops.rbln_custom_ops, op.get_attn_op_name(), fake_op, raising=False)
    op.phase = phase
    q = torch.zeros(1, op.num_key_value_heads, op.num_heads // op.num_key_value_heads, 1, op.head_dim)
    kv = torch.zeros(1, op.num_key_value_heads, 1, op.head_dim)
    cache = torch.zeros(1, op.num_key_value_heads, 1, op.head_dim)
    seq = (torch.zeros(1, 1, dtype=torch.int32), torch.zeros(1, 1, dtype=torch.int32))
    op(q, kv, kv, torch.ones(1, 16), cache, cache, seq, torch.tensor(1.0), torch.zeros(1, 1, dtype=torch.int16), 16)
    return captured["is_bidirectional"]


@pytest.fixture
def gemma4_layers():
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=16,
    )
    return Gemma4ForCausalLM(config).model.layers


@pytest.mark.parametrize("attn_impl", ["eager", "flash_attn"])
def test_gemma4_full_layers_stay_causal_in_image_prefill(monkeypatch, gemma4_layers, attn_impl):
    rbln_config = RBLNGemma4ForCausalLMConfig(
        max_seq_len=64, sliding_window=16, attn_impl=attn_impl, phases=["prefill", "image_prefill", "decode"]
    )
    rbln_config.kvcache_block_size = 16 if attn_impl == "flash_attn" else 64
    rbln_config.kvcache_partition_len = 16 if attn_impl == "flash_attn" else None

    sliding = Gemma4TextAttention(gemma4_layers[0].self_attn, rbln_config, is_sliding=True).get_attention_op()
    full = Gemma4TextAttention(gemma4_layers[1].self_attn, rbln_config, is_sliding=False).get_attention_op()

    assert sliding.bidirectional_image_prefill is True
    assert full.bidirectional_image_prefill is False
    assert _capture_is_bidirectional(monkeypatch, full, "image_prefill") is False
    assert _capture_is_bidirectional(monkeypatch, full, "prefill") is False


def test_gemma3_full_layers_keep_bidirectional_image_prefill(monkeypatch):
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention"],
        sliding_window=16,
    )
    rbln_config = RBLNGemma3ForCausalLMConfig(
        max_seq_len=64, sliding_window=16, attn_impl="eager", phases=["prefill", "image_prefill", "decode"]
    )
    rbln_config.kvcache_block_size = 64
    self_attn = Gemma3ForCausalLM(config).model.layers[0].self_attn
    full = Gemma3Attention(self_attn, rbln_config, is_sliding=False).get_attention_op()

    assert _capture_is_bidirectional(monkeypatch, full, "image_prefill") is True
    assert _capture_is_bidirectional(monkeypatch, full, "prefill") is False
