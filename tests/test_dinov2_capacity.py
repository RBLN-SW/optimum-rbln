"""Native518 capacity regressions for the dedicated real CR13 scheduler lane."""

import os

import pytest

from . import test_transformers as model_cases


pytestmark = pytest.mark.skipif(
    os.environ.get("RBLN_TEST_NPU") != "RBLN-CR13",
    reason="Native batch1024 capacity requires the dedicated CR13 scheduler lane.",
)


class TestDinov2ModelNativeSizeBatch1024CR13(model_cases.TestDinov2ModelNativeSizeBatch512):
    BATCH_SIZE = 1024
    RBLN_CLASS_KWARGS = {
        **model_cases.TestDinov2ModelNativeSizeBatch512.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 1024,
        "rbln_npu": "RBLN-CR13",
        "rbln_encoder_chunk_size": None,
    }


class TestDinov2ForImageClassificationNativeSizeBatch1024CR13(
    model_cases.TestDinov2ForImageClassificationNativeSizeBatch512
):
    BATCH_SIZE = 1024
    RBLN_CLASS_KWARGS = TestDinov2ModelNativeSizeBatch1024CR13.RBLN_CLASS_KWARGS
