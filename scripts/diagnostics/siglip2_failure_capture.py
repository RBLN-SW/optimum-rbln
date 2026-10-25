"""Observe original parity failures after execution without modifying model calls."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch


def _tensor_record(value):
    assert isinstance(value, torch.Tensor)
    assert value.device.type == "cpu"
    value = value.detach().contiguous().clone()
    return value, {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
    }


def capture_frame(local, root):
    instance = local["self"]
    case = type(instance).__name__
    folder = Path(root) / case
    folder.mkdir(parents=True, exist_ok=False)
    payload = {}
    receipt = {
        "test_class": case,
        "offset": int(local["offset"]),
        "batch_size": int(instance.BATCH_SIZE),
        "original_failures": local["failures"],
        "scope": "Post-failure observer; original inputs, graph, calls and assertions unchanged",
        "tensors": {},
    }
    for key in ("actual", "expected", "repeated"):
        assert len(local[key]) == 8
        values, records = zip(*(_tensor_record(value) for value in local[key]), strict=True)
        payload[key] = tuple(values)
        receipt["tensors"][key] = list(records)
    payload["inputs"] = {}
    receipt["tensors"]["inputs"] = {}
    for key, value in local["inputs"].items():
        payload["inputs"][key], receipt["tensors"]["inputs"][key] = _tensor_record(value)
    path = folder / "original-parity.pt"
    torch.save(payload, path)
    with path.open("rb") as stream:
        receipt["artifact_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt["artifact_bytes"] = path.stat().st_size
    (folder / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        "SIGLIP2_FAILURE_CAPTURE "
        + json.dumps(
            {
                "test_class": case,
                "offset": receipt["offset"],
                "artifact_bytes": receipt["artifact_bytes"],
                "artifact_sha256": receipt["artifact_sha256"],
            }
        ),
        flush=True,
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when != "call" or not report.failed or call.excinfo is None:
        return
    if (
        item.name != "test_numerical_parity"
        or item.cls is None
        or not item.cls.__name__.startswith("TestSiglip2Model")
    ):
        return
    for entry in call.excinfo.traceback:
        if entry.name != "test_numerical_parity":
            continue
        local = entry.frame.f_locals
        if {"actual", "expected", "repeated", "inputs", "failures", "offset", "self"} <= local.keys():
            capture_frame(local, os.environ["SIGLIP2_FAILURE_CAPTURE_DIR"])
            return
    raise AssertionError("Original SigLIP2 parity failure frame was not captured")
