import os
import unittest

import pytest
import rebel


_REAL_COMPILER = rebel.compile_from_torch


class TestDefaultCompileMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.check_mode()

    @classmethod
    def check_mode(cls):
        assert (rebel.compile_from_torch is _REAL_COMPILER) == (os.environ.get("OPTIMUM_RBLN_REAL_COMPILE") == "1")

    def test_mode(self):
        self.check_mode()

    @classmethod
    def tearDownClass(cls):
        cls.check_mode()


@pytest.mark.requires_compile
class TestRequiredCompileMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert rebel.compile_from_torch is _REAL_COMPILER

    def test_mode(self):
        assert rebel.compile_from_torch is _REAL_COMPILER
        from tests.fake_rbln import is_fake_compile

        assert not is_fake_compile()

    @classmethod
    def tearDownClass(cls):
        assert rebel.compile_from_torch is _REAL_COMPILER


class TestInheritedRequiredCompileMode(TestRequiredCompileMode):
    pass


class TestDefaultCompileModeAfterReal(TestDefaultCompileMode):
    pass
