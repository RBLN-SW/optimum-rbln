import json

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


def test_prefix_lm_is_not_serialized_and_is_redeclared_on_load(tmp_path):
    config = RBLNDecoderOnlyModelForCausalLMConfig(use_attention_mask=True, use_position_ids=True)
    config._is_prefix_lm = True
    serialized = config._prepare_for_serialization()
    assert "_is_prefix_lm" not in serialized

    (tmp_path / "rbln_config.json").write_text(json.dumps(serialized))
    assert RBLNDecoderOnlyModelForCausalLMConfig.from_pretrained(tmp_path)._is_prefix_lm is False
    # The passed instance carries it across the load, as the parent's load hook relies on.
    assert RBLNDecoderOnlyModelForCausalLMConfig.from_pretrained(tmp_path, rbln_config=config)._is_prefix_lm is True


@pytest.mark.parametrize("parent_cls", [RBLNPaliGemmaForConditionalGeneration, RBLNPaliGemmaModel])
def test_paligemma_declares_prefix_lm_only_for_the_language_model(parent_cls):
    lm = RBLNDecoderOnlyModelForCausalLMConfig(use_attention_mask=True, use_position_ids=True)
    assert parent_cls._update_submodule_rbln_config_on_load("language_model", lm)._is_prefix_lm is True
    vision = RBLNDecoderOnlyModelForCausalLMConfig()
    assert parent_cls._update_submodule_rbln_config_on_load("vision_tower", vision)._is_prefix_lm is False
