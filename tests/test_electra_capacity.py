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


class TestElectraModelBatch32CR13(_CapacityAllocation, encoder_cases.TestElectraModel):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 32
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraModel.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 32,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(32, 1) for name, tensor in encoder_cases.TestElectraModel.GENERATION_KWARGS.items()
    }


class TestElectraForPreTrainingBatch32CR13(_CapacityAllocation, encoder_cases.TestElectraForPreTraining):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 32
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraForPreTraining.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 32,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(32, 1)
        for name, tensor in encoder_cases.TestElectraForPreTraining.GENERATION_KWARGS.items()
    }


class TestElectraForSequenceClassificationBatch32CR13(
    _CapacityAllocation, encoder_cases.TestElectraForSequenceClassification
):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 32
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraForSequenceClassification.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 32,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(32, 1)
        for name, tensor in encoder_cases.TestElectraForSequenceClassification.GENERATION_KWARGS.items()
    }


class TestElectraModelBatch64CR13(_CapacityAllocation, encoder_cases.TestElectraModel):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 64
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraModel.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 64,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(64, 1) for name, tensor in encoder_cases.TestElectraModel.GENERATION_KWARGS.items()
    }


class TestElectraForPreTrainingBatch64CR13(_CapacityAllocation, encoder_cases.TestElectraForPreTraining):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 64
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraForPreTraining.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 64,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(64, 1)
        for name, tensor in encoder_cases.TestElectraForPreTraining.GENERATION_KWARGS.items()
    }


class TestElectraForSequenceClassificationBatch64CR13(
    _CapacityAllocation, encoder_cases.TestElectraForSequenceClassification
):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 64
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraForSequenceClassification.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 64,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(64, 1)
        for name, tensor in encoder_cases.TestElectraForSequenceClassification.GENERATION_KWARGS.items()
    }


class TestElectraModelBatch128CR13(_CapacityAllocation, encoder_cases.TestElectraModel):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 128
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraModel.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 128,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(128, 1) for name, tensor in encoder_cases.TestElectraModel.GENERATION_KWARGS.items()
    }


class TestElectraForPreTrainingBatch128CR13(_CapacityAllocation, encoder_cases.TestElectraForPreTraining):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 128
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraForPreTraining.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 128,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(128, 1)
        for name, tensor in encoder_cases.TestElectraForPreTraining.GENERATION_KWARGS.items()
    }


class TestElectraForSequenceClassificationBatch128CR13(
    _CapacityAllocation, encoder_cases.TestElectraForSequenceClassification
):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 128
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraForSequenceClassification.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 128,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(128, 1)
        for name, tensor in encoder_cases.TestElectraForSequenceClassification.GENERATION_KWARGS.items()
    }


class TestElectraModelBatch256CR13(_CapacityAllocation, encoder_cases.TestElectraModel):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 256
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraModel.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 256,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(256, 1) for name, tensor in encoder_cases.TestElectraModel.GENERATION_KWARGS.items()
    }


class TestElectraForPreTrainingBatch256CR13(_CapacityAllocation, encoder_cases.TestElectraForPreTraining):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 256
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraForPreTraining.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 256,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(256, 1)
        for name, tensor in encoder_cases.TestElectraForPreTraining.GENERATION_KWARGS.items()
    }


class TestElectraForSequenceClassificationBatch256CR13(
    _CapacityAllocation, encoder_cases.TestElectraForSequenceClassification
):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 256
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraForSequenceClassification.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 256,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(256, 1)
        for name, tensor in encoder_cases.TestElectraForSequenceClassification.GENERATION_KWARGS.items()
    }


class TestElectraModelBatch512CR13(_CapacityAllocation, encoder_cases.TestElectraModel):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 512
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraModel.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 512,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(512, 1) for name, tensor in encoder_cases.TestElectraModel.GENERATION_KWARGS.items()
    }


class TestElectraForPreTrainingBatch512CR13(_CapacityAllocation, encoder_cases.TestElectraForPreTraining):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 512
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraForPreTraining.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 512,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(512, 1)
        for name, tensor in encoder_cases.TestElectraForPreTraining.GENERATION_KWARGS.items()
    }


class TestElectraForSequenceClassificationBatch512CR13(
    _CapacityAllocation, encoder_cases.TestElectraForSequenceClassification
):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 512
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraForSequenceClassification.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 512,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(512, 1)
        for name, tensor in encoder_cases.TestElectraForSequenceClassification.GENERATION_KWARGS.items()
    }


class TestElectraModelBatch1024CR13(_CapacityAllocation, encoder_cases.TestElectraModel):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 1024
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraModel.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 1024,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(1024, 1) for name, tensor in encoder_cases.TestElectraModel.GENERATION_KWARGS.items()
    }


class TestElectraForPreTrainingBatch1024CR13(_CapacityAllocation, encoder_cases.TestElectraForPreTraining):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 1024
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraForPreTraining.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 1024,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(1024, 1)
        for name, tensor in encoder_cases.TestElectraForPreTraining.GENERATION_KWARGS.items()
    }


class TestElectraForSequenceClassificationBatch1024CR13(
    _CapacityAllocation, encoder_cases.TestElectraForSequenceClassification
):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 1024
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraForSequenceClassification.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 1024,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(1024, 1)
        for name, tensor in encoder_cases.TestElectraForSequenceClassification.GENERATION_KWARGS.items()
    }


class TestElectraModelBatch2048CR13(_CapacityAllocation, encoder_cases.TestElectraModel):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 2048
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraModel.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 2048,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(2048, 1) for name, tensor in encoder_cases.TestElectraModel.GENERATION_KWARGS.items()
    }


class TestElectraForPreTrainingBatch2048CR13(_CapacityAllocation, encoder_cases.TestElectraForPreTraining):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 2048
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraForPreTraining.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 2048,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(2048, 1)
        for name, tensor in encoder_cases.TestElectraForPreTraining.GENERATION_KWARGS.items()
    }


class TestElectraForSequenceClassificationBatch2048CR13(
    _CapacityAllocation, encoder_cases.TestElectraForSequenceClassification
):
    TEST_LEVEL = encoder_cases.TestLevel.FULL
    BATCH_SIZE = 2048
    RBLN_CLASS_KWARGS = {
        **encoder_cases.TestElectraForSequenceClassification.RBLN_CLASS_KWARGS,
        "rbln_batch_size": 2048,
        "rbln_npu": "RBLN-CR13",
    }
    GENERATION_KWARGS = {
        name: tensor.repeat(2048, 1)
        for name, tensor in encoder_cases.TestElectraForSequenceClassification.GENERATION_KWARGS.items()
    }
