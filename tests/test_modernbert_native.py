"""Native 8192-token ModernBERT powers of two on the CR13 scheduler lane."""

import os

import pytest

from . import test_transformers


pytestmark = pytest.mark.skipif(
    os.environ.get("RBLN_TEST_NPU") != "RBLN-CR13", reason="Larger native batches are validated in the CR13 lane"
)


class TestModernBertModelNativeMaxSeqLenBatch2(test_transformers.TestModernBertModelNativeMaxSeqLen):
    BATCH_SIZE = 2
    RBLN_CLASS_KWARGS = {
        **test_transformers.TestModernBertModelNativeMaxSeqLen.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 2,
        "rbln_npu": "RBLN-CR13",
    }


class TestModernBertModelNativeMaxSeqLenBatch4(test_transformers.TestModernBertModelNativeMaxSeqLen):
    BATCH_SIZE = 4
    RBLN_CLASS_KWARGS = {
        **test_transformers.TestModernBertModelNativeMaxSeqLen.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 4,
        "rbln_npu": "RBLN-CR13",
    }


class TestModernBertModelNativeMaxSeqLenBatch8(test_transformers.TestModernBertModelNativeMaxSeqLen):
    BATCH_SIZE = 8
    RBLN_CLASS_KWARGS = {
        **test_transformers.TestModernBertModelNativeMaxSeqLen.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 8,
        "rbln_npu": "RBLN-CR13",
    }


class TestModernBertModelNativeMaxSeqLenBatch16(test_transformers.TestModernBertModelNativeMaxSeqLen):
    BATCH_SIZE = 16
    RBLN_CLASS_KWARGS = {
        **test_transformers.TestModernBertModelNativeMaxSeqLen.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 16,
        "rbln_npu": "RBLN-CR13",
    }
