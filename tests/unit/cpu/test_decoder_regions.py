from types import SimpleNamespace

import pytest
import rebel
import torch
from rebel.core.compilation._impl import clone_nn_module
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
@pytest.mark.parametrize("max_seq_len", [8192, 16384, 40960])
def test_decoder_export(monkeypatch, enabled, phase, gemma, max_seq_len):
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
        max_position_embeddings=max_seq_len,
    )
    model = hf_model_cls(cfg).eval()
    config = config_cls(
        batch_size=1,
        max_seq_len=max_seq_len,
        npu="RBLN-CR13",
        kvcache_partition_len=8192 if max_seq_len > 8192 else None,
        prefill_chunk_size=128,
        kvcache_num_blocks=max_seq_len // 8192,
        num_devices=1,
        create_runtimes=False,
        dtype="float32",
        nested_compile_region=enabled,
    )
    config = model_cls._update_rbln_config(model=model, model_config=cfg, rbln_config=config)
    assert config.requires_batch_sort is (max_seq_len > 8192)
    wrapped = model_cls._wrap_model_if_needed(model, config)
    ccfg = config.compile_cfgs[0 if phase == "prefill" else 1]
    inputs = ccfg.get_dummy_inputs(
        fill=0, meta_tensor_names=[n for n, _, _ in ccfg.input_info if "past_key_values" in n]
    )
    context, _ = model_cls._get_compile_context(ccfg, inputs)

    def export(model, **kwargs):
        assert kwargs.get("model_trace_method") == ("export_strict" if enabled else None)
        result = torch.export.export(clone_nn_module(model), tuple(kwargs["example_inputs"]), strict=enabled)
        assert set(result.graph_signature.inputs_to_parameters.values()) <= set(
            dict(model.named_parameters(remove_duplicate=False))
        )
        calls = [n for n in result.graph.nodes if "invoke_subgraph" in str(n.target)]
        assert len(calls) == ((4 if gemma else 3) if enabled else 0)
        if enabled:
            assert calls[0].args[1] == calls[2].args[1]
            assert (calls[0].args[1] != calls[1].args[1]) is gemma
            if gemma:
                assert calls[1].args[1] == calls[3].args[1]
        if config.requires_batch_sort:
            attention_nodes = [
                node
                for module in result.graph_module.modules()
                if isinstance(module, torch.fx.GraphModule)
                for node in module.graph.nodes
                if "paged" in str(node.target) and "attn" in str(node.target)
            ]
            assert attention_nodes
            for node in attention_nodes:
                arguments = {
                    arg.name: value for arg, value in zip(node.target._schema.arguments, node.args, strict=False)
                }
                arguments.update(node.kwargs)
                if any(arg.name == "is_prompt_sorted" for arg in node.target._schema.arguments):
                    assert arguments["is_prompt_sorted"] is True
                else:
                    assert "is_prompt_sorted" not in arguments
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


@pytest.mark.parametrize("supports_sort", [False, True])
def test_schema_query_inside_region(supports_sort):
    from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import _op_accepts

    library = torch.library.Library("rbln_custom_ops", "FRAGMENT")
    argument = ", bool is_prompt_sorted=False" if supports_sort else ""
    library.define(f"test_decoder_region_schema(Tensor input{argument}) -> Tensor")

    @torch.compiler.nested_compile_region
    def region(x):
        if _op_accepts("test_decoder_region_schema", "is_prompt_sorted"):
            return x + 1
        return x - 1

    class Model(torch.nn.Module):
        def forward(self, x):
            return region(region(x))

    inputs = (torch.zeros(2),)
    exported = torch.export.export(Model(), inputs, strict=True)
    torch.testing.assert_close(exported.module()(*inputs), torch.full((2,), 2.0 if supports_sort else -2.0))
