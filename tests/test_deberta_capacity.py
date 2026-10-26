"""Native512 capacity rows, executed only by the dedicated owned CR13 lane."""

import os

import pytest

from . import test_transformers as encoder_cases


pytestmark = pytest.mark.skipif(
    os.environ.get("RBLN_TEST_NPU") != "RBLN-CR13",
    reason="Native512 capacity cases require the dedicated CR13 scheduler lane.",
)


class TestDebertaV3ModelBatch32CR13(encoder_cases.TestDebertaV3Model):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 32
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestDebertaV3Model.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 32,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(32, 1) for name, tensor in encoder_cases.TestDebertaV3Model.GENERATION_KWARGS.items()
    }


class TestDebertaV3ForSequenceClassificationBatch32CR13(encoder_cases.TestDebertaV3ForSequenceClassification):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 32
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestDebertaV3ForSequenceClassification.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 32,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(32, 1)
        for name, tensor in encoder_cases.TestDebertaV3ForSequenceClassification.GENERATION_KWARGS.items()
    }


class TestDebertaV3ModelBatch64CR13(encoder_cases.TestDebertaV3Model):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 64
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestDebertaV3Model.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 64,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(64, 1) for name, tensor in encoder_cases.TestDebertaV3Model.GENERATION_KWARGS.items()
    }


class TestDebertaV3ForSequenceClassificationBatch64CR13(encoder_cases.TestDebertaV3ForSequenceClassification):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 64
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestDebertaV3ForSequenceClassification.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 64,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(64, 1)
        for name, tensor in encoder_cases.TestDebertaV3ForSequenceClassification.GENERATION_KWARGS.items()
    }
