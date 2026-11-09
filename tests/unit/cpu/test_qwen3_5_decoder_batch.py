import pytest

from optimum.rbln.transformers.models.qwen3_5.configuration_qwen3_5 import (
    RBLNQwen3_5ForCausalLMConfig,
    RBLNQwen3_5ForConditionalGenerationConfig,
)


GENERATION_CONFIGS = [RBLNQwen3_5ForCausalLMConfig, RBLNQwen3_5ForConditionalGenerationConfig]


@pytest.mark.parametrize("config_cls", GENERATION_CONFIGS)
@pytest.mark.parametrize("decoder_batch_sizes", [None, [4]])
def test_one_decoder_batch_size_is_the_default(config_cls, decoder_batch_sizes):
    # Unset or a single entry is what every Qwen3.5 model compiles with, so neither is refused.
    config = config_cls(max_seq_len=4096, batch_size=4, decoder_batch_sizes=decoder_batch_sizes)
    assert config.decoder_batch_sizes == [4]


@pytest.mark.parametrize("config_cls", GENERATION_CONFIGS)
@pytest.mark.parametrize("decoder_batch_sizes", [[4, 2], [4, 2, 1]])
def test_decoder_batch_bucketing_is_refused_while_the_config_is_built(config_cls, decoder_batch_sizes):
    # The linear_attention state is addressed by batch row, so only the full-batch decoder can run.
    # Refused here rather than at `_update_rbln_config`, which a vision model only reaches once it
    # has compiled its visual submodule -- so the error has to come before that, not after.
    with pytest.raises(ValueError, match="one decoder batch size"):
        config_cls(max_seq_len=4096, batch_size=4, decoder_batch_sizes=decoder_batch_sizes)
