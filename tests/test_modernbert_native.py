"""Native 8192-token ModernBERT powers of two on the CR13 scheduler lane."""

import os

import pytest

from . import test_transformers


pytestmark = pytest.mark.skipif(
    os.environ.get("RBLN_TEST_NPU") != "RBLN-CR13", reason="Larger native batches are validated in the CR13 lane"
)


class TestModernBertModelNativeBatches:
    @pytest.mark.parametrize("batch_size", [2, 4, 8, 16])
    def test_embedding_parity(self, batch_size):
        case = type(
            f"ModernBertModelNativeMaxSeqLenBatch{batch_size}",
            (test_transformers.TestModernBertModelNativeMaxSeqLen,),
            {
                "BATCH_SIZE": batch_size,
                "RBLN_CLASS_KWARGS": {
                    **test_transformers.TestModernBertModelNativeMaxSeqLen.RBLN_CLASS_KWARGS,
                    "rbln_batch_size": batch_size,
                    "rbln_npu": "RBLN-CR13",
                },
            },
        )
        try:
            case.setUpClass()
            case("test_embedding_parity").test_embedding_parity()
        finally:
            case.tearDownClass()
