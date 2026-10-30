import json
import operator
from types import SimpleNamespace

import pytest
import rebel
import torch
from rebel.core.compilation._impl import clone_nn_module
from transformers import (
    Gemma2Config,
    Gemma2ForCausalLM,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from optimum.rbln import (
    RBLNGemma2ForCausalLM,
    RBLNGemma2ForCausalLMConfig,
    RBLNQwen2_5_VLForConditionalGeneration,
    RBLNQwen2_5_VLForConditionalGenerationConfig,
    RBLNQwen3_5ForConditionalGeneration,
    RBLNQwen3_5ForConditionalGenerationConfig,
    RBLNQwen3_5ModelConfig,
    RBLNQwen3ForCausalLM,
    RBLNQwen3ForCausalLMConfig,
    RBLNQwen3VLForConditionalGenerationConfig,
    RBLNQwen3VLModelConfig,
)
from optimum.rbln.transformers.models.gpt_oss.gpt_oss_architecture import RBLNGptOssExperts


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    ("config_cls", "vision_cls_name"),
    [
        (RBLNQwen3ForCausalLMConfig, None),
        (RBLNQwen3VLForConditionalGenerationConfig, "RBLNQwen3VLVisionModelConfig"),
        (RBLNQwen3VLModelConfig, "RBLNQwen3VLVisionModelConfig"),
        (RBLNQwen3_5ForConditionalGenerationConfig, "RBLNQwen3_5VisionModelConfig"),
        (RBLNQwen3_5ModelConfig, "RBLNQwen3_5VisionModelConfig"),
    ],
)
def test_region_config_not_saved(config_cls, vision_cls_name, enabled, tmp_path):
    kwargs = (
        {"use_inputs_embeds": True, "visual": {"cls_name": vision_cls_name, "max_seq_len": 64}}
        if vision_cls_name
        else {}
    )
    config = config_cls(nested_compile_region=enabled, **kwargs)
    assert config.nested_compile_region is enabled
    config.save(tmp_path)
    assert "nested_compile_region" not in json.loads((tmp_path / "rbln_config.json").read_text())
    assert config_cls.from_pretrained(tmp_path).nested_compile_region is False
    restored = config_cls.from_pretrained(tmp_path, rbln_nested_compile_region=enabled)
    assert restored.nested_compile_region is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_region_config_legacy_override(enabled, tmp_path):
    RBLNQwen3ForCausalLMConfig().save(tmp_path)
    path = tmp_path / "rbln_config.json"
    saved = json.loads(path.read_text())
    saved["nested_compile_region"] = enabled
    path.write_text(json.dumps(saved))
    restored = RBLNQwen3ForCausalLMConfig.from_pretrained(tmp_path, rbln_nested_compile_region=not enabled)
    assert restored.nested_compile_region is not enabled


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


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("hybrid", [False, True], ids=["qwen2_5_vl", "qwen3_5"])
def test_packed_rotary_decoder_export(enabled, phase, hybrid):
    text = {
        "vocab_size": 128,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 4 if hybrid else 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 16384,
    }
    vision = {"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 128}
    extra = {}
    if hybrid:
        text.update(
            head_dim=64,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
        )
        vision["num_position_embeddings"] = 64
        hf_config_cls, hf_model_cls = Qwen3_5Config, Qwen3_5ForConditionalGeneration
        model_cls = RBLNQwen3_5ForConditionalGeneration
        config_cls = RBLNQwen3_5ForConditionalGenerationConfig
        vision_cls = "RBLNQwen3_5VisionModelConfig"
        extra["gdn_chunk_size"] = 128
    else:
        text["rope_parameters"] = {"rope_type": "default", "rope_theta": 10000, "mrope_section": [8, 12, 12]}
        vision["fullatt_block_indexes"] = [0]
        hf_config_cls, hf_model_cls = Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
        model_cls = RBLNQwen2_5_VLForConditionalGeneration
        config_cls = RBLNQwen2_5_VLForConditionalGenerationConfig
        vision_cls = "RBLNQwen2_5_VisionTransformerPretrainedModelConfig"
    cfg = hf_config_cls(text_config=text, vision_config=vision)
    model = hf_model_cls(cfg).eval()
    config = config_cls(
        batch_size=1,
        max_seq_len=16384,
        npu="RBLN-CR13",
        kvcache_partition_len=8192,
        prefill_chunk_size=128,
        kvcache_num_blocks=2,
        num_devices=1,
        create_runtimes=False,
        dtype="float32",
        nested_compile_region=enabled,
        visual={"cls_name": vision_cls, "max_seq_len": 64},
        **extra,
    )
    config = model_cls._update_rbln_config(model=model, model_config=cfg, rbln_config=config)
    wrapped = model_cls._wrap_model_if_needed(model, config)
    wrapped.phase = phase
    ccfg = config.compile_cfgs[0 if phase == "prefill" else 1]
    inputs = ccfg.get_dummy_inputs(
        fill=0, meta_tensor_names=[n for n, _, _ in ccfg.input_info if "past_key_values" in n]
    )
    # Keep Qwen3.5's single full-attention region visible on newer PyTorch.
    dynamo_options = (
        {"inline_single_use_invoke_subgraph": False}
        if hasattr(torch._dynamo.config, "inline_single_use_invoke_subgraph")
        else {}
    )
    with torch.inference_mode(), torch._dynamo.config.patch(**dynamo_options):
        exported = torch.export.export(clone_nn_module(wrapped), tuple(inputs), strict=enabled)
    nodes = list(exported.graph.nodes)
    calls = [n for n in nodes if "invoke_subgraph" in str(n.target)]
    assert len(calls) == ((4 if hybrid else 3) if enabled else 0)
    if enabled:
        # The compiler rolls layer calls into a loop only when nothing runs between them.
        between = [
            n
            for n in nodes[nodes.index(calls[0]) : nodes.index(calls[-1])]
            if n.op == "call_function" and n not in calls and not (n.target is operator.getitem and n.args[0] in calls)
        ]
        assert not between


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
