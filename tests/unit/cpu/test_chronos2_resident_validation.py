"""Resident graph boundaries preserve masks and bind fresh runtime objects."""

from types import SimpleNamespace

import pytest
import torch

from scripts.diagnostics.chronos2_frozen_validation import NAMES
from scripts.diagnostics.chronos2_resident_validation import ResidentRuntimeChain, resident_adapter


def test_resident_boundaries_preserve_all_hidden_rows_and_both_masks():
    hidden = torch.arange(24.0).reshape(2, 3, 4)
    time_mask = torch.zeros(2, 1, 1, 3)
    group_mask = torch.ones(3, 1, 2, 2)
    calls = []

    def encoder(value, time, group):
        assert value.shape == hidden.shape
        assert torch.equal(time, time_mask)
        assert torch.equal(group, group_mask)
        assert value.data_ptr() != hidden.data_ptr()
        calls.append(1)
        return [value + 1]

    chain = ResidentRuntimeChain(
        [lambda *args: (hidden, time_mask, group_mask), *([encoder] * 4), lambda value: (value, value * 2)],
        lambda value: None,
    )
    first, residual = chain(None)
    assert len(calls) == 4
    torch.testing.assert_close(first, hidden + 4, atol=0, rtol=0)
    torch.testing.assert_close(residual, (hidden + 4) * 2, atol=0, rtol=0)


def test_reload_uses_only_the_passed_compiled_models():
    created = []

    class Compiled:
        def create_runtime(self, **kwargs):
            created.append(self)
            assert kwargs["device"] == 7
            return object()

    config = SimpleNamespace(
        compile_cfgs=[SimpleNamespace(compiled_model_name=n) for n in NAMES],
        device_map=dict.fromkeys(NAMES, 7),
        activate_profiler=False,
        timeout=30,
    )
    adapter = resident_adapter({}, lambda value: None)
    first, fresh = [Compiled() for _ in NAMES], [Compiled() for _ in NAMES]
    a = adapter(None, first, config)[0]
    b = adapter(None, fresh, config)[0]
    assert created == first + fresh
    assert all(x is not y for x, y in zip(a.runtimes, b.runtimes, strict=True))
    config.device_map["forecast"] = 8
    with pytest.raises(ValueError, match="same scheduler-owned"):
        adapter(None, fresh, config)
