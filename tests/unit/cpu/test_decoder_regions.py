from types import SimpleNamespace

import pytest
import rebel
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

from optimum.rbln import (
    RBLNGemma2ForCausalLM,
    RBLNGemma2ForCausalLMConfig,
    RBLNQwen3ForCausalLM,
    RBLNQwen3ForCausalLMConfig,
)
from optimum.rbln.transformers.models.gpt_oss.gpt_oss_architecture import RBLNGptOssExperts


@pytest.mark.parametrize("enabled", [False, True])
def test_region_config_roundtrip(enabled, tmp_path):
    config = RBLNQwen3ForCausalLMConfig(nested_compile_region=enabled)
    config.save(tmp_path)
    restored = RBLNQwen3ForCausalLMConfig.from_pretrained(tmp_path)
    assert restored.nested_compile_region is enabled
    assert RBLNQwen3ForCausalLMConfig().nested_compile_region is False


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("gemma", [False, True])
def test_decoder_export(monkeypatch, enabled, phase, gemma):
    hf_config_cls, hf_model_cls, model_cls, config_cls = (
        (Gemma2Config, Gemma2ForCausalLM, RBLNGemma2ForCausalLM, RBLNGemma2ForCausalLMConfig)
        if gemma
        else (Qwen3Config, Qwen3ForCausalLM, RBLNQwen3ForCausalLM, RBLNQwen3ForCausalLMConfig)
    )
    cfg = hf_config_cls(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4 if gemma else 3,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=8192,
    )
    model = hf_model_cls(cfg).eval()
    config = config_cls(
        batch_size=1,
        max_seq_len=8192,
        prefill_chunk_size=128,
        kvcache_num_blocks=1,
        num_devices=1,
        create_runtimes=False,
        dtype="float32",
        nested_compile_region=enabled,
    )
    config = model_cls._update_rbln_config(model=model, model_config=cfg, rbln_config=config)
    wrapped = model_cls._wrap_model_if_needed(model, config)
    ccfg = config.compile_cfgs[0 if phase == "prefill" else 1]
    inputs = ccfg.get_dummy_inputs(
        fill=0, meta_tensor_names=[n for n, _, _ in ccfg.input_info if "past_key_values" in n]
    )
    context, _ = model_cls._get_compile_context(ccfg, inputs)

    def export(model, **kwargs):
        assert kwargs.get("model_trace_method") == ("export_strict" if enabled else None)
        result = torch.export.export(model, tuple(kwargs["example_inputs"]), strict=enabled)
        calls = [n for n in result.graph.nodes if "invoke_subgraph" in str(n.target)]
        assert len(calls) == ((4 if gemma else 3) if enabled else 0)
        if enabled:
            assert calls[0].args[1] == calls[2].args[1]
            assert (calls[0].args[1] != calls[1].args[1]) is gemma
            if gemma:
                assert calls[1].args[1] == calls[3].args[1]
        return result

    monkeypatch.setattr(rebel, "compile_from_torch", export)
    with torch.inference_mode():
        model_cls._compile_model(wrapped, ccfg, inputs, context, config, phase=phase)


def test_gpt_oss_experts_have_nonaliased_inputs():
    model = SimpleNamespace(
        num_experts=2,
        intermediate_size=64,
        hidden_size=64,
        alpha=1.702,
        limit=7.0,
        gate_up_proj_blocks=torch.randint(256, (2, 128, 2, 16), dtype=torch.uint8),
        gate_up_proj_scales=torch.randint(256, (2, 128, 2), dtype=torch.uint8),
        gate_up_proj_bias=torch.randn(2, 128),
        down_proj_blocks=torch.randint(256, (2, 64, 2, 16), dtype=torch.uint8),
        down_proj_scales=torch.randint(256, (2, 64, 2), dtype=torch.uint8),
        down_proj_bias=torch.randn(2, 64),
    )
    experts = RBLNGptOssExperts(model, top_k=1)
    for suffix in ("blocks", "scales", "bias"):
        gate = getattr(experts, f"gate_proj_{suffix}")
        up = getattr(experts, f"up_proj_{suffix}")
        assert gate.untyped_storage().data_ptr() != up.untyped_storage().data_ptr()
