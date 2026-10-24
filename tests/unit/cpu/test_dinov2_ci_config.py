import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(("target", "chunk"), [(None, 6), ("RBLN-CA25", 6), ("RBLN-CR13", None)])
def test_native_batch512_configuration_matches_ci_target(target, chunk):
    environment = dict(os.environ)
    environment.pop("RBLN_TEST_NPU", None)
    if target is not None:
        environment["RBLN_TEST_NPU"] = target
    source = """
import json
from tests.test_transformers import (
    TestDinov2ModelNativeSizeBatch512 as Encoder,
    TestDinov2ForImageClassificationNativeSizeBatch512 as Classifier,
)
assert Encoder.RBLN_CLASS_KWARGS == Classifier.RBLN_CLASS_KWARGS
print('DINO_CI_CONFIG ' + json.dumps(Encoder.RBLN_CLASS_KWARGS))
"""
    process = subprocess.run(
        [sys.executable, "-c", source], env=environment, capture_output=True, text=True, check=True
    )
    line = next(line for line in process.stdout.splitlines() if line.startswith("DINO_CI_CONFIG "))
    configuration = json.loads(line.removeprefix("DINO_CI_CONFIG "))
    assert configuration["rbln_encoder_chunk_size"] == chunk
    assert configuration["rbln_batch_size"] == 512
    assert configuration["rbln_image_size"] == 518
