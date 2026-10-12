from unittest.mock import Mock

import pytest
import torch

from optimum.rbln.transformers.models.decoderonly.configuration_decoderonly import (
    RBLNDecoderOnlyModelForCausalLMConfig,
)


def config(**kwargs):
    return RBLNDecoderOnlyModelForCausalLMConfig(
        max_seq_len=8192, prefill_chunk_size=8192, use_attention_mask=True, use_position_ids=True, **kwargs
    )


def test_bucket_names_keep_legacy_prefill_and_decoder_names():
    cfg = config(prefill_chunk_sizes=[512, 1024, 2048, 4096, 8192])
    assert cfg.expected_compiled_model_names == [
        "prefill",
        "prefill_512",
        "prefill_1024",
        "prefill_2048",
        "prefill_4096",
        "decoder_batch_1",
    ]
    assert cfg.decoder_runtime_idx == 5
    assert cfg.max_seq_len == cfg.prefill_chunk_size == 8192


def test_old_saved_config_keeps_two_runtime_names():
    cfg = config()
    assert cfg.expected_compiled_model_names == ["prefill", "decoder_batch_1"]
    assert cfg.decoder_runtime_idx == 1


@pytest.mark.parametrize("sizes", [[0], [513], [16384], [512, 512], [True]])
def test_invalid_bucket_sizes_are_rejected(sizes):
    with pytest.raises(ValueError, match="prefill_chunk_sizes"):
        config(prefill_chunk_sizes=sizes)


@pytest.mark.parametrize("length,expected", [(260, 512), (512, 512), (513, 1024), (1025, 8192), (8192, 8192)])
def test_whole_bidirectional_prompt_is_sent_to_one_bucket(length, expected):
    from optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils import BidirectionalPrefillBuckets

    runtimes = {n: Mock(return_value=n) for n in [512, 1024, 8192]}
    dispatcher = BidirectionalPrefillBuckets(runtimes)
    inputs = torch.zeros(1, length, 8)
    assert dispatcher(inputs_embeds=inputs) == expected
    runtimes[expected].assert_called_once_with(inputs_embeds=inputs)
    assert sum(rt.call_count for rt in runtimes.values()) == 1


def test_long_prompt_is_not_split_to_fit_short_bucket():
    from optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils import BidirectionalPrefillBuckets

    dispatcher = BidirectionalPrefillBuckets({512: Mock()})
    with pytest.raises(ValueError, match="512"):
        dispatcher(input_ids=torch.zeros(1, 513, dtype=torch.long))


def test_prefix_cache_keeps_original_largest_bucket_contract():
    from optimum.rbln.transformers.models.decoderonly.decoderonly_runtime_utils import BidirectionalPrefillBuckets

    runtimes = {512: Mock(), 8192: Mock(return_value="largest")}
    dispatcher = BidirectionalPrefillBuckets(runtimes)
    inputs = torch.zeros(1, 260, 8)
    cache_position = torch.arange(8192, 8192 + 260).unsqueeze(0)
    assert dispatcher(inputs_embeds=inputs, cache_position=cache_position) == "largest"
    runtimes[512].assert_not_called()


def test_bucket_compile_shapes_and_decoder_capacity():
    from transformers import GemmaConfig, GemmaForCausalLM

    from optimum.rbln import RBLNGemmaForCausalLM, RBLNGemmaForCausalLMConfig

    hf_config = GemmaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    model = GemmaForCausalLM(hf_config).eval()
    cfg = RBLNGemmaForCausalLMConfig(
        max_seq_len=8192,
        prefill_chunk_size=8192,
        use_attention_mask=True,
        use_position_ids=True,
        use_inputs_embeds=True,
    )
    cfg = RBLNGemmaForCausalLM._update_rbln_config(None, model, hf_config, cfg)
    assert cfg.extra_prefill_chunk_sizes == [512, 1024, 2048, 4096]
    assert [c.compiled_model_name for c in cfg.compile_cfgs] == cfg.expected_compiled_model_names
    for compile_cfg, length in zip(cfg.compile_cfgs[:5], [8192, 512, 1024, 2048, 4096], strict=True):
        profile = {name: shape for name, shape, _ in compile_cfg.input_info}
        assert profile["inputs_embeds"][1] == length
        assert profile["attention_mask"][-1] == 8192
    cfg.freeze()
    from optimum.rbln.transformers.models.decoderonly.modeling_decoderonly import _prefill_bucket_config

    view = _prefill_bucket_config(cfg, 512)
    assert cfg.prefill_chunk_size == 8192
    assert view.prefill_chunk_size == 512
    assert view.max_seq_len == 8192


def test_causal_prefill_does_not_compile_bidirectional_buckets():
    from transformers import GemmaConfig, GemmaForCausalLM

    from optimum.rbln import RBLNGemmaForCausalLM, RBLNGemmaForCausalLMConfig

    hf_config = GemmaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    model = GemmaForCausalLM(hf_config).eval()
    cfg = RBLNGemmaForCausalLMConfig(
        max_seq_len=8192, prefill_chunk_size=512, use_attention_mask=False, use_position_ids=False
    )
    cfg = RBLNGemmaForCausalLM._update_rbln_config(None, model, hf_config, cfg)
    assert cfg.extra_prefill_chunk_sizes == []
    assert cfg.expected_compiled_model_names == ["prefill", "decoder_batch_1"]


def test_bucket_compiles_preserve_parameter_storage_for_weight_sharing():
    from unittest.mock import patch

    from transformers import GemmaConfig, GemmaForCausalLM

    from optimum.rbln import RBLNGemmaForCausalLM, RBLNGemmaForCausalLMConfig

    hf = GemmaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    model = GemmaForCausalLM(hf).eval()
    cfg = RBLNGemmaForCausalLMConfig(
        max_seq_len=1024,
        prefill_chunk_size=1024,
        use_attention_mask=True,
        use_position_ids=True,
        use_inputs_embeds=True,
    )
    cfg = RBLNGemmaForCausalLM._update_rbln_config(None, model, hf, cfg)
    calls = []

    def compile_graph(wrapped, compile_cfg, inputs, context, view, quantization=None, phase="prefill"):
        calls.append(
            (
                wrapped,
                tuple(p.untyped_storage()._cdata for p in wrapped.parameters()),
                wrapped.rbln_config.prefill_chunk_size,
                context,
                inputs,
            )
        )
        assert all(getattr(m, "rbln_config", view) is view for m in wrapped.modules())
        return Mock()

    with patch.object(RBLNGemmaForCausalLM, "_compile_model", side_effect=compile_graph):
        compiled = RBLNGemmaForCausalLM.get_compiled_model(model, cfg)
    assert list(compiled) == cfg.expected_compiled_model_names
    assert [c[2] for c in calls] == [1024, 512, 1024]
    assert all(c[0] is calls[0][0] and c[1] == calls[0][1] and c[3] is calls[0][3] for c in calls)
    assert calls[0][0].rbln_config is cfg
