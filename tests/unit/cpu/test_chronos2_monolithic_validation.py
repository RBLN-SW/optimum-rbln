"""The monolithic suite must use the product's original runtime factory."""

import sys
import unittest
from types import SimpleNamespace

from scripts.diagnostics import chronos2_frozen_validation as original


def test_original_runtime_factory_reaches_monolithic_suite_unchanged(monkeypatch, tmp_path):
    from optimum.rbln import RBLNChronos2Model

    factory = RBLNChronos2Model._create_runtimes.__func__
    fake_class = type("OriginalTest", (unittest.TestCase,), {})
    monkeypatch.setitem(sys.modules, "tests.test_transformers", SimpleNamespace(TestChronos2ModelBatch512=fake_class))
    monkeypatch.setattr(unittest.defaultTestLoader, "getTestCaseNames", lambda cls: [str(i) for i in range(9)])
    monkeypatch.setattr(unittest.defaultTestLoader, "loadTestsFromTestCase", lambda cls: object())

    def reject_adapter(*args, **kwargs):
        raise AssertionError("The native path must not install a sequential adapter")

    def execute(runner, suite):
        assert RBLNChronos2Model._create_runtimes.__func__ is factory
        return SimpleNamespace(
            testsRun=9,
            skipped=[("automap", "RBLN_AUTO_CLASS is None")],
            failures=[],
            errors=[],
            wasSuccessful=lambda: True,
        )

    monkeypatch.setattr(original, "runtime_adapter", reject_adapter)
    monkeypatch.setattr(unittest.TextTestRunner, "run", execute)
    events = []
    original.run_suite(tmp_path, {}, tmp_path, events.append, sequential=False)
    assert events[-1]["phase"] == "original_suite"
    assert events[-1]["passed"]
    assert RBLNChronos2Model._create_runtimes.__func__ is factory
