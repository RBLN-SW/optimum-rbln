import pytest

from optimum.rbln.transformers.models.decoderonly.configuration_decoderonly import (
    RBLNDecoderOnlyModelForCausalLMConfig,
)
from optimum.rbln.transformers.models.gemma3.configuration_gemma3 import RBLNGemma3ForCausalLMConfig
from optimum.rbln.transformers.models.gemma4.configuration_gemma4 import RBLNGemma4ForCausalLMConfig
from optimum.rbln.transformers.models.paligemma.modeling_paligemma import (
    RBLNPaliGemmaForConditionalGeneration,
    RBLNPaliGemmaModel,
)


@pytest.mark.parametrize("config_cls", [RBLNGemma3ForCausalLMConfig, RBLNGemma4ForCausalLMConfig])
def test_standalone_gemma_text_prefill_is_causal(config_cls):
    config = config_cls()
    assert config.use_attention_mask
    assert config.use_position_ids
    assert config._is_prefix_lm is False


def test_mask_and_position_ids_alone_do_not_imply_prefix_lm():
    config = RBLNDecoderOnlyModelForCausalLMConfig(use_attention_mask=True, use_position_ids=True)
    assert config._is_prefix_lm is False


def test_prefix_lm_survives_the_rbln_config_round_trip():
    config = RBLNDecoderOnlyModelForCausalLMConfig(use_attention_mask=True, use_position_ids=True)
    config._is_prefix_lm = True
    reloaded = RBLNDecoderOnlyModelForCausalLMConfig(**config._prepare_for_serialization())
    assert reloaded._is_prefix_lm is True


@pytest.mark.parametrize("parent_cls", [RBLNPaliGemmaForConditionalGeneration, RBLNPaliGemmaModel])
def test_paligemma_declares_prefix_lm_for_its_language_model(parent_cls):
    hf_config = type("HFConfig", (), {})()
    lm = RBLNDecoderOnlyModelForCausalLMConfig(use_attention_mask=True, use_position_ids=True)
    lm = parent_cls._update_submodule_rbln_config("language_model", None, None, hf_config, lm, None)
    assert lm._is_prefix_lm is True
    assert hf_config.use_sliding_window is False

    vision = RBLNDecoderOnlyModelForCausalLMConfig()
    assert parent_cls._update_submodule_rbln_config("vision_tower", None, None, hf_config, vision, None) is vision
    assert vision._is_prefix_lm is False


@pytest.mark.parametrize("parent_cls", [RBLNPaliGemmaForConditionalGeneration, RBLNPaliGemmaModel])
def test_paligemma_rejects_a_language_model_without_mask_and_position_ids(parent_cls):
    lm = RBLNDecoderOnlyModelForCausalLMConfig(use_attention_mask=True)
    with pytest.raises(ValueError, match="use_position_ids"):
        parent_cls._update_submodule_rbln_config("language_model", None, None, type("HFConfig", (), {})(), lm, None)
