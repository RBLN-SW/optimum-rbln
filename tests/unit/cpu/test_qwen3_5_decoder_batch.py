from types import SimpleNamespace

import pytest
import torch

from optimum.rbln.transformers.models.qwen3_5.configuration_qwen3_5 import (
    RBLNQwen3_5ForCausalLMConfig,
    RBLNQwen3_5ForConditionalGenerationConfig,
)
from optimum.rbln.transformers.models.qwen3_5.qwen3_5_runtime_utils import RBLNQwen3_5RuntimeModel


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


def _decode_runtime(batch_size, *, requires_batch_sort=False):
    """A decode runtime with no device: `_run` records the inputs the graph would take."""
    runtime = object.__new__(RBLNQwen3_5RuntimeModel)
    runtime.rbln_config = SimpleNamespace(
        use_lora=False,
        use_attention_mask=False,
        use_inputs_embeds=True,
        requires_batch_sort=requires_batch_sort,
    )
    runtime.batch_size = batch_size
    runtime.phase = "decode"
    runtime._is_legacy_graph = False
    runtime._decode_state_idx = torch.arange(batch_size, dtype=torch.int16)
    runtime.graph_inputs = {}
    runtime._run = lambda named: (runtime.graph_inputs.update(named), (torch.zeros(batch_size, 1, 8), None))[1]
    return runtime


def _decode(runtime, cache_position, state_idx=None):
    inputs = torch.zeros(runtime.batch_size, 1, 8)
    return runtime.decode_forward(inputs, torch.tensor(cache_position).reshape(-1, 1), state_idx=state_idx)


def test_decode_rows_are_the_batch_rows_unless_the_caller_names_them():
    # optimum's own generate keeps decode row b on cache row b.
    runtime = _decode_runtime(4)
    _decode(runtime, [9, 7, 5, 3])
    assert runtime.graph_inputs["state_idx"].tolist() == [0, 1, 2, 3]

    # vLLM lays the batch out in its own order and names each row's cache row.
    rows = torch.tensor([2, 0, 3, 1], dtype=torch.int16)
    _decode(runtime, [9, 7, 5, 3], state_idx=rows)
    assert runtime.graph_inputs["state_idx"] is rows


def test_decode_batch_sorted_by_length_when_compiled_for_it():
    # The in-memory flash decode kernel takes the rows longest first.
    runtime = _decode_runtime(2, requires_batch_sort=True)
    _decode(runtime, [9, 5])
    with pytest.raises(ValueError, match="requires_batch_sort"):
        _decode(runtime, [5, 9])


def test_prefill_takes_its_row_as_batch_idx_not_state_idx():
    runtime = _decode_runtime(1)
    runtime.phase = "prefill"
    with pytest.raises(ValueError, match="decode input"):
        runtime.forward(inputs_embeds=torch.zeros(1, 4, 8), state_idx=torch.zeros(1, dtype=torch.int16))
