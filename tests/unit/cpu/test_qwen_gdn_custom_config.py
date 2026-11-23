from types import SimpleNamespace

import pytest

from optimum.rbln.transformers.models.qwen3_5 import modeling_qwen3_5


def test_gdn_ops_need_a_compiler_that_registers_them(monkeypatch):
    # The core always runs as the ops, so a compiler without them is reported while the config is
    # built, by the names it misses -- not at compile time, as a lookup that finds nothing. The
    # compiler this suite runs on is not asked: which ops it registers is its own business.
    empty = SimpleNamespace(ops=SimpleNamespace(rbln_custom_ops=object()))
    monkeypatch.setattr(modeling_qwen3_5, "torch", empty)
    with pytest.raises(ValueError, match="chunk_gated_delta_rule"):
        modeling_qwen3_5.validate_gdn_ops()
