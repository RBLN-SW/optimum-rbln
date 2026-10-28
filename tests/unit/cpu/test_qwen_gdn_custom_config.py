import re
from types import SimpleNamespace

import pytest
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from optimum.rbln.configuration_utils import RBLNCompileConfig
from optimum.rbln.transformers.models.qwen3_5 import modeling_qwen3_5
from optimum.rbln.transformers.models.qwen3_5.configuration_qwen3_5 import (
    RBLNQwen3_5TextModelConfig,
)
from optimum.rbln.transformers.models.qwen3_5.modeling_qwen3_5 import (
    _qwen3_5_legacy_linear_state_shapes,
    _qwen3_5_linear_state_shapes,
    uses_custom_kernel,
    validate_gdn_custom_kernel,
    warn_deprecated,
)


def test_gdn_custom_kernel_needs_a_compiler_that_registers_the_ops(monkeypatch):
    # The core always runs as the ops, so a compiler without them is reported while the config is
    # built, by the names it misses -- not at compile time, as a lookup that finds nothing. The
    # compiler this suite runs on is not asked: which ops it registers is its own business.
    empty = SimpleNamespace(ops=SimpleNamespace(rbln_custom_ops=object()))
    monkeypatch.setattr(modeling_qwen3_5, "torch", empty)
    with pytest.raises(ValueError, match="chunk_gated_delta_rule"):
        validate_gdn_custom_kernel()


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
    assert uses_custom_kernel(legacy) is False

    current = _config_with_graph_inputs("input_ids", "state_mask", "valid_mask", "conv_hist_idx")
    assert uses_custom_kernel(current) is True


def test_the_graph_path_keeps_its_own_cache_layouts():
    # The runtime sizes that graph's control masks from here, so these are the layouts the artifact
    # was compiled with: the conv cache channel-last, the recurrent state 3D.
    text_config = Qwen3_5TextConfig(
        hidden_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    taps = text_config.linear_conv_kernel_dim - 1
    assert _qwen3_5_legacy_linear_state_shapes(text_config, 1) == ((1, taps, 2 * 2 * 128 + 6 * 128), (1, 6 * 128, 128))
    # and they are not the ones the kernels take.
    assert _qwen3_5_legacy_linear_state_shapes(text_config, 1) != _qwen3_5_linear_state_shapes(text_config, 1)


def test_a_deprecated_path_warns_until_its_version_and_then_refuses():
    # Loading such an artifact says so, and at the version that drops the path it stops loading.
    warn_deprecated("That path", version="99.0.0", remedy="Recompile the model.")
    with pytest.raises(ValueError, match=re.escape("That path is deprecated and was removed in version 0.0.1.")):
        warn_deprecated("That path", version="0.0.1")
