"""Native512 capacity rows, executed only by the dedicated owned CR13 lane."""

import json
import os
from pathlib import Path

import pytest
import rebel

from . import test_transformers as encoder_cases


pytestmark = pytest.mark.skipif(
    os.environ.get("RBLN_TEST_NPU") != "RBLN-CR13",
    reason="Native512 capacity cases require the dedicated CR13 scheduler lane.",
)


class _CapacityAllocation:
    @classmethod
    def setUpClass(cls):
        print(f"CAPACITY_SETUP {cls.__name__}: original compile/runtime setup starting", flush=True)
        super().setUpClass()
        for artifact in sorted(Path(cls.get_rbln_local_dir()).glob("*.rbln")):
            metadata = rebel.RBLNCompiledModel.inspect(artifact)
            print(
                "CAPACITY_ALLOCATION "
                + json.dumps(
                    {
                        "test_class": cls.__name__,
                        "artifact": str(artifact),
                        "uuid": metadata["uuid"],
                        "npu": metadata["npu"],
                        "npu_aggregate_allocation_bytes": metadata["alloc_per_node"],
                        "note": "Aggregate per NPU; not a per-chiplet allocation or free-memory reading.",
                    }
                ),
                flush=True,
            )


class TestDebertaV3ModelBatch32CR13(_CapacityAllocation, encoder_cases.TestDebertaV3Model):
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


class TestDebertaV3ForSequenceClassificationBatch32CR13(
    _CapacityAllocation, encoder_cases.TestDebertaV3ForSequenceClassification
):
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


class TestDebertaV3ModelBatch64CR13(_CapacityAllocation, encoder_cases.TestDebertaV3Model):
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


class TestDebertaV3ForSequenceClassificationBatch64CR13(
    _CapacityAllocation, encoder_cases.TestDebertaV3ForSequenceClassification
):
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
