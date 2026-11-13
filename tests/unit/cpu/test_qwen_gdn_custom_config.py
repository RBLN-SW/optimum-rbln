from types import SimpleNamespace

import pytest

from optimum.rbln.configuration_utils import RBLNCompileConfig
from optimum.rbln.transformers.models.qwen3_5 import modeling_qwen3_5
from optimum.rbln.transformers.models.qwen3_5.configuration_qwen3_5 import (
    RBLNQwen3_5TextModelConfig,
)
from optimum.rbln.transformers.models.qwen3_5.modeling_qwen3_5 import (
    use_custom_ops,
    validate_gdn_ops,
)


def test_gdn_ops_need_a_compiler_that_registers_them(monkeypatch):
    # The core always runs as the ops, so a compiler without them is reported while the config is
    # built, by the names it misses -- not at compile time, as a lookup that finds nothing. The
    # compiler this suite runs on is not asked: which ops it registers is its own business.
    empty = SimpleNamespace(ops=SimpleNamespace(rbln_custom_ops=object()))
    monkeypatch.setattr(modeling_qwen3_5, "torch", empty)
    with pytest.raises(ValueError, match="chunk_gated_delta_rule"):
        validate_gdn_ops()


def _config_with_graph_inputs(*names):
    config = RBLNQwen3_5TextModelConfig(max_seq_len=4096, batch_size=1)
    config.set_compile_cfgs(
        [RBLNCompileConfig(compiled_model_name="prefill", input_info=[(n, [1], "float32") for n in names])]
    )
    return config


def test_a_pre_kernel_artifact_is_recognized_by_its_own_graph():
    # Nothing in the config marks one: a config written before the kernels could not have written
    # anything about them. The graph's inputs are what answers, and they differ either way.
    legacy = _config_with_graph_inputs("input_ids", "conv_state_mask", "recurrent_state_mask", "valid_mask")
    assert use_custom_ops(legacy) is False

    current = _config_with_graph_inputs("input_ids", "state_mask", "valid_mask", "state_idx")
    assert use_custom_ops(current) is True
