from optimum.rbln.transformers.models.decoderonly.configuration_decoderonly import (
    RBLNDecoderOnlyModelForCausalLMConfig,
)
from optimum.rbln.transformers.models.gemma3.configuration_gemma3 import RBLNGemma3ForCausalLMConfig


def test_standalone_gemma3_text_prefill_is_causal():
    config = RBLNGemma3ForCausalLMConfig()
    assert config.use_attention_mask
    assert config.use_position_ids
    assert not config.use_image_prefill
    assert config.use_bidirectional_prefill is False


def test_prefix_lm_prefill_stays_bidirectional_for_other_decoders():
    config = RBLNDecoderOnlyModelForCausalLMConfig(use_attention_mask=True, use_position_ids=True)
    assert config.use_bidirectional_prefill is True
