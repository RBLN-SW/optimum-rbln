"""Run unchanged native Chronos batch classes with the exact PR14913 official SDK."""

import argparse
import base64
import contextlib
import hashlib
import importlib.metadata
import json
import os
import tempfile
import unittest
from functools import wraps
from pathlib import Path
from unittest.mock import patch


SDK = "0.12.1.dev343+g8f59b810"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
CHECKPOINT_DIGEST = "6b8714cb88218fd3a9b96b8104d49aa14891cc22511858e9db714ea3df579355"


def official_sdk_receipt():
    from tvm._ffi import base as ffi_base

    from scripts.diagnostics.chronos2_frozen_validation import sha256

    distribution = importlib.metadata.distribution("rebel-compiler")
    if distribution.version != SDK:
        raise ValueError("Official SDK version differs from the reviewed head")
    library = Path(ffi_base._LIB._name).resolve()
    records = [f for f in distribution.files if Path(str(f)).name == "librbln.so"]
    record = next((f for f in records if Path(distribution.locate_file(f)).resolve() == library), None)
    if record is None or record.hash is None or record.hash.mode != "sha256":
        raise ValueError("Loaded compiler DSO is not bound to the installed wheel RECORD")
    digest = sha256(library)
    encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip("=")
    if encoded != record.hash.value:
        raise ValueError("Loaded compiler DSO bytes differ from the official wheel RECORD")
    return {"version": SDK, "DSO_sha256": digest, "wheel_RECORD_matches_loaded_DSO": True}


def checkpoint_receipt():
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from scripts.diagnostics.chronos2_prediction_audit import tensor_sha

    checkpoint = hf_hub_download("amazon/chronos-2", "model.safetensors", revision=REVISION)
    weights = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as stream:
        tensor_names = stream.keys()
        for name in tensor_names:
            value = stream.get_tensor(name)
            weights[name] = {"shape": list(value.shape), "sha256": tensor_sha(value)}
    digest = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != CHECKPOINT_DIGEST:
        raise ValueError("Original frozen checkpoint tensor hashes changed")
    if len(weights) != 170:
        raise ValueError("Original native checkpoint tensor membership changed")
    return weights


def run_class(cls, npu, root, weights, event):
    import rebel
    from chronos import Chronos2Model

    from optimum.rbln import RBLNChronos2Model
    from optimum.rbln.utils.runtime_utils import ContextRblnConfig
    from scripts.diagnostics.chronos2_frozen_validation import sha256
    from scripts.diagnostics.chronos2_prediction_audit import PredictionAudit

    methods = unittest.defaultTestLoader.getTestCaseNames(cls)
    if len(methods) != 9 or cls.HF_CONFIG_KWARGS["revision"] != REVISION:
        raise ValueError("Original native test selection/checkpoint revision changed")
    batch = cls.BATCH_SIZE
    audit = PredictionAudit(root / "predictions", weights)
    original_setup, original_teardown = cls.setUpClass, cls.tearDownClass
    native_factory = RBLNChronos2Model._create_runtimes.__func__
    artifacts = []

    def setup(test_cls):
        original_setup()
        model = test_cls.model
        if (model.rbln_config.batch_size, model.context_length, model.prediction_length) != (batch, 8192, 1024):
            raise ValueError("Native maximum geometry changed")
        if len(model.compiled_models) != 1 or RBLNChronos2Model._create_runtimes.__func__ is not native_factory:
            raise ValueError("Native original model must remain one graph with its original runtime factory")
        path = Path(test_cls.get_rbln_local_dir()) / "compiled_model.rbln"
        metadata = rebel.RBLNCompiledModel.inspect(path)
        if metadata["compiler_version"] != SDK or metadata["npu"] != npu or metadata["num_devices"] != 1:
            raise ValueError("Actual compiled artifact SDK/hardware differs from this job")
        if len(metadata["outputs"]) != 2 or any(
            r["shape"] != [batch, 21, 1024] or r["dtype"] != "float32" for r in metadata["outputs"]
        ):
            raise ValueError("Original FP32 forecast/residual tuple changed")
        artifacts.append(
            {
                "class": test_cls.__name__,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "metadata": metadata,
                "alloc_per_node": model.compiled_models[0].get_alloc_per_node(),
            }
        )
        (root / "compiled-artifact.json").write_text(json.dumps(artifacts[-1], indent=2) + "\n")
        event(
            {
                "phase": "native_compiled",
                "class": test_cls.__name__,
                "batch": batch,
                "artifact_sha256": artifacts[-1]["sha256"],
                "alloc_per_node": artifacts[-1]["alloc_per_node"],
            }
        )

    def teardown(test_cls):
        try:
            with tempfile.TemporaryDirectory(dir=root) as temporary:
                test_cls.model.save_pretrained(temporary)
                if sha256(Path(temporary) / "compiled_model.rbln") != artifacts[-1]["sha256"]:
                    raise ValueError("Saved official artifact bytes changed")
                prior = test_cls.model.compiled_models
                with ContextRblnConfig(device=0):
                    fresh = test_cls.RBLN_CLASS.from_pretrained(temporary, export=False)
                if any(a is b for a, b in zip(prior, fresh.compiled_models, strict=True)):
                    raise ValueError("Fresh official reload reused original compiled-model objects")
                test_cls.model = fresh
                test_cls("test_numerical_parity").test_numerical_parity()
                event({"phase": "fresh_reload_original_parity", "class": test_cls.__name__, "passed": True})
        finally:
            original_teardown()

    cpu_forward, product_forward = Chronos2Model.forward, RBLNChronos2Model.forward

    @wraps(cpu_forward)
    def cpu_capture(model, *args, **kwargs):
        inputs = {**kwargs, **({"context": args[0]} if args else {})}
        output = cpu_forward(model, *args, **kwargs)
        audit.reference(model, inputs, output)
        return output

    @wraps(product_forward)
    def product_capture(model, *args, **kwargs):
        inputs = {**kwargs, **({"context": args[0]} if args else {})}
        with audit.product_request(inputs):
            output = product_forward(model, *args, **kwargs)
            audit.product_output(output)
            return output

    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(cls, "setUpClass", classmethod(setup)))
            stack.enter_context(patch.object(cls, "tearDownClass", classmethod(teardown)))
            stack.enter_context(patch.object(Chronos2Model, "forward", cpu_capture))
            stack.enter_context(patch.object(RBLNChronos2Model, "forward", product_capture))
            result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    finally:
        prediction_report = audit.finish()
    record = {
        "class": cls.__name__,
        "batch": batch,
        "methods": methods,
        "tests_run": result.testsRun,
        "skipped": [(str(t), reason) for t, reason in result.skipped],
        "failures": len(result.failures),
        "errors": len(result.errors),
        "original_suite_passed": result.wasSuccessful(),
        "all_series_passed": prediction_report["all_series_passed"],
        "identical_input_outputs_exact": prediction_report["identical_input_outputs_exact"],
        "reference_calls": len(prediction_report["reference_calls"]),
        "matched_calls": prediction_report["matched_calls"],
        "unique_reference_inputs": prediction_report["unique_reference_inputs"],
    }
    expected_unique = 4 if batch <= 2 else 5
    record["accepted"] = bool(
        result.wasSuccessful()
        and result.testsRun == 9
        and len(result.skipped) == 1
        and "RBLN_AUTO_CLASS is None" in result.skipped[0][1]
        and record["all_series_passed"]
        and record["identical_input_outputs_exact"]
        and record["reference_calls"] == 10
        and record["matched_calls"] == 22
        and record["unique_reference_inputs"] == expected_unique
    )
    (root / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    event({"phase": "original_native_suite", **record})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", choices=BATCHES, required=True)
    parser.add_argument("--npu", choices=["RBLN-CA25", "RBLN-CR13"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("BUILDKITE_JOB_ID") or os.environ.get("RBLN_TEST_NPU") != args.npu:
        raise ValueError("Official matrix runs only on its scheduler-owned target")
    if len(set(args.batches)) != len(args.batches) or not args.batches:
        raise ValueError("Native batch selection must be explicit, nonempty and unique")
    if any(
        k in os.environ
        for k in ["SAVE_ARTIFACTS_PATH", "REUSE_ARTIFACTS_PATH", "TVM_LIBRARY_PATH", "RBLN_DUMMY_DEVICE"]
    ):
        raise ValueError("Artifact compatibility, dummy and library override modes are forbidden")
    os.environ["OPTIMUM_RBLN_TEST_LEVEL"] = "full"
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    events = args.out / "events.jsonl"

    def event(value):
        with events.open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print("CHRONOS_OFFICIAL " + json.dumps(value), flush=True)

    sdk = official_sdk_receipt()
    weights = checkpoint_receipt()
    (args.out / "sdk.json").write_text(json.dumps(sdk, indent=2) + "\n")
    (args.out / "checkpoint-weights.json").write_text(json.dumps(weights, indent=2) + "\n")
    event({"phase": "official_preflight", "SDK": sdk, "batches": args.batches, "npu": args.npu})
    from tests import test_transformers

    records = []
    for batch in args.batches:
        name = "TestChronos2Model" if batch == 1 else "TestChronos2ModelBatch" + str(batch)
        cls = getattr(test_transformers, name)
        directory = args.out / name
        directory.mkdir(exist_ok=False)
        os.chdir(directory)
        records.append(run_class(cls, args.npu, directory, weights, event))
        (args.out / "result.json").write_text(
            json.dumps({"SDK": sdk, "npu": args.npu, "classes": records}, indent=2) + "\n"
        )
    if len(records) != len(args.batches) or not all(row["accepted"] for row in records):
        raise RuntimeError("Official native batch matrix has incomplete/failed acceptance rows")


if __name__ == "__main__":
    main()
