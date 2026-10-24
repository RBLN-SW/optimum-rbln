"""Run the unchanged native B512 suite against six frozen, sequential CA25 jobs."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from unittest.mock import patch

from scripts.diagnostics.chronos2_sequential_runtime import SequentialRuntimeChain


NAMES = ("embedding", "encoder_0", "encoder_3", "encoder_6", "encoder_9", "forecast")
SOURCE = "4b1449d15e3630fae8b1b86b52dc5d3b90eb85b6"
STAGED_SHA = "6da3a379773884765fa5fc8d38aeeae52a7deb76f5192336949c4a5effc5f0a0"
PRIMITIVE = "ffb04afbbf16627a93805257542b201b9fa3afb8"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_frozen(root, manifest_sha256, inspect_model=None, manifest_name="manifest.json"):
    """Fail before any runtime on incomplete, altered, or reduced-geometry data."""
    root = Path(root)
    if Path(manifest_name).name != manifest_name:
        raise ValueError("Manifest name must be a file within the frozen root")
    if sha256(root / manifest_name) != manifest_sha256:
        raise ValueError("Frozen manifest SHA256 mismatch")
    manifest = json.loads((root / manifest_name).read_text())
    expected = {
        "status": "compiled",
        "compile_only": True,
        "runtime_executed": False,
        "source_compiler": SOURCE,
        "source_optimum": "bbc20bad08c895d6179128cb9efe8862c85726ba",
        "frontend_sha256": "522ccfdcaf665165ad9f2e8a4a45a53a9d92d2b4c6a766529aa4545db78e8e1e",
        "mapped_librbln_sha256": STAGED_SHA,
        "primitive": PRIMITIVE,
        "checkpoint_revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
        "batch_size": 512,
        "context_length": 8192,
        "prediction_length": 1024,
        "encoder_chunk_size": 3,
        "frozen_inputs_exact_reused": True,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Frozen manifest mismatch: {key}")
    for name in ("config.json", "original_rbln_config.json", "compile_configs.json"):
        receipt = manifest["files"][name]
        path = root / name
        if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
            raise ValueError("Frozen source config checksum/size mismatch")
    if (root / "rbln_config.json").exists():
        cloned = json.loads((root / "rbln_config.json").read_text())
        if cloned["_compile_cfgs"] != manifest["compile_configs"]:
            raise ValueError("Cloned six-stage config changed")
    provenance = manifest["staged_logging_provenance"]
    if (
        provenance["source_base"] != "5f544486937380881f229e48ed9b2161f243e265"
        or provenance["logging_diff_sha256"] != "8ad4826b3ddef7408374354c005f5d415077b951cf913876bcd61f6272e6af70"
    ):
        raise ValueError("Reviewed staged source provenance mismatch")
    if manifest["activation"]["gelu_module_count"] != 0:
        raise ValueError("Expected unchanged native ReLU/no GELU graph")
    if sha256(root / "original-test-generate-inputs.pt") != manifest["original_input_file_sha256"]:
        raise ValueError("Frozen native inputs changed")
    stages = manifest["stages"]
    if [stage["name"] for stage in stages] != list(NAMES):
        raise ValueError("Expected all six native stages")
    hidden, time_mask, group_mask = [512, 577, 768], [512, 1, 1, 577], [577, 1, 512, 512]
    original = json.loads((root / "original_rbln_config.json").read_text())
    if (original["batch_size"], original["context_length"], original["prediction_length"]) != (512, 8192, 1024):
        raise ValueError("Original full native geometry changed")
    for stage, cfg in zip(stages, manifest["compile_configs"], strict=True):
        path = root / stage["file"]
        if stage["file"] != stage["name"] + ".rbln" or stage["status"] != "compiled":
            raise ValueError("Invalid frozen stage binding")
        if path.stat().st_size != stage["bytes"] or sha256(path) != stage["sha256"]:
            raise ValueError("Frozen stage SHA256/size mismatch")
        metadata = stage["metadata"]
        if inspect_model is not None:
            actual = inspect_model(path)
            for key in ("inputs", "outputs", "npu", "num_devices", "subgraph"):
                if actual[key] != metadata[key]:
                    raise ValueError(f"Actual compiled interface mismatch: {stage['name']} {key}")
        if metadata["npu"] != "RBLN-CA25" or metadata["num_devices"] != 1:
            raise ValueError("Expected one CA25")
        if metadata["subgraph"]["stats"] != [{"compile_id": 0, "device_func_count": 1, "host_op_reason": {}}]:
            raise ValueError("Unexpected reported fallback/device-function summary")
        if cfg["compiled_model_name"] != stage["name"]:
            raise ValueError("Stage/config name mismatch")
        actual_inputs = [[item["name"], item["shape"], item["dtype"]] for item in metadata["inputs"]]
        if actual_inputs != cfg["input_info"]:
            raise ValueError("Config input interface mismatch")
        shapes = [item["shape"] for item in metadata["outputs"]]
        if stage["name"] == "embedding":
            expected_inputs = original["_compile_cfgs"][0]["input_info"]
            expected_outputs = [hidden, time_mask, group_mask]
        elif stage["name"].startswith("encoder_"):
            expected_inputs = [
                ["hidden", hidden, "float32"],
                ["attention_mask", time_mask, "float32"],
                ["group_time_mask", group_mask, "float32"],
            ]
            expected_outputs = [hidden]
        else:
            expected_inputs = [["hidden", hidden, "float32"]]
            expected_outputs = [[512, 21, 1024], [512, 21, 1024]]
        if actual_inputs != expected_inputs or shapes != expected_outputs:
            raise ValueError("Full joint native geometry/forecast tuple mismatch")
        if any(item["dtype"] != "float32" for item in metadata["outputs"]):
            raise ValueError("Expected original FP32 boundary outputs")
    return manifest


def block_export(*args, **kwargs):
    raise RuntimeError("Frozen diagnostic forbids model export/compilation")


def runtime_adapter(manifest, event, observer=None):
    """The closure must use each invocation's passed models, including fresh reloads."""

    def create(cls, compiled_models, rbln_config):
        if len(compiled_models) != len(NAMES) or [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs] != list(
            NAMES
        ):
            raise ValueError("Expected six passed compiled-model objects")
        models = dict(zip(NAMES, compiled_models, strict=True))
        devices = [rbln_config.device_map[name] for name in NAMES]
        if any(device != devices[0] for device in devices):
            raise ValueError("All six stages must use the same CA25")

        @contextmanager
        def loader(path, device):
            runtime = models[path.stem].create_runtime(
                tensor_type="pt",
                device=device,
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            try:
                yield runtime
            finally:
                del runtime
                gc.collect()

        event({"phase": "runtime_chain_created", "compiled_model_object_ids": [id(x) for x in compiled_models]})
        return [
            SequentialRuntimeChain(Path("."), manifest["stages"], devices[0], event, loader=loader, observer=observer)
        ]

    return create


def stage_bundle(root, target, manifest):
    """Clone the original product configuration, changing only six compile interfaces."""
    from optimum.rbln import RBLNChronos2ModelConfig
    from optimum.rbln.configuration_utils import RBLNCompileConfig

    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "config.json", target / "config.json")
    for stage in manifest["stages"]:
        shutil.copyfile(root / stage["file"], target / stage["file"])
    config = RBLNChronos2ModelConfig.from_pretrained(root / "original_rbln_config.json")
    # set_compile_cfgs propagates these global values over every per-stage config.
    config.npu = manifest["compile_configs"][0]["npu"]
    config.num_devices = manifest["compile_configs"][0]["num_devices"]
    config.set_compile_cfgs([RBLNCompileConfig(**cfg) for cfg in manifest["compile_configs"]])
    config.save(target / "rbln_config.json")


def verify_runtime_library(expected_sha):
    import rebel

    version = importlib.metadata.version("rebel-compiler")
    if version != "0.12.1.dev329+g66a3078e":
        raise ValueError(f"Expected published SDK329 runtime, got {version}")
    mapped = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "/librbln.so" in line}
    if len(mapped) != 1 or sha256(next(iter(mapped))) != expected_sha:
        raise ValueError("Expected exactly the pinned SDK329 library mapping")
    return {"version": version, "library": next(iter(mapped)), "sha256": expected_sha, "rebel_module": rebel.__file__}


def run_suite(root, manifest, work, event, diagnostics=None):
    from optimum.rbln import RBLNChronos2Model
    from optimum.rbln.utils.runtime_utils import ContextRblnConfig
    from tests.test_transformers import TestChronos2ModelBatch512

    cls = TestChronos2ModelBatch512
    methods = unittest.defaultTestLoader.getTestCaseNames(cls)
    if len(methods) != 9:
        raise ValueError(f"Original native test selection changed: {methods}")
    original_teardown = cls.tearDownClass

    def setup(test_cls):
        local = Path(test_cls.get_rbln_local_dir())
        if local.exists():
            raise ValueError("Refusing to overwrite existing native-test artifact directory")
        stage_bundle(root, local, manifest)
        with ContextRblnConfig(device=0):
            test_cls.model = test_cls.RBLN_CLASS.from_pretrained(local, export=False)

    def teardown(test_cls):
        try:
            # Original save/load methods are unchanged. This additional original
            # parity invocation validates inference from freshly reloaded objects.
            with tempfile.TemporaryDirectory(dir=work) as temporary:
                test_cls.model.save_pretrained(temporary)
                for stage in manifest["stages"]:
                    if sha256(Path(temporary) / stage["file"]) != stage["sha256"]:
                        raise ValueError("Saved artifact identity changed")
                prior = test_cls.model.compiled_models
                with ContextRblnConfig(device=0):
                    fresh = test_cls.RBLN_CLASS.from_pretrained(temporary, export=False)
                if any(a is b for a, b in zip(prior, fresh.compiled_models, strict=True)):
                    raise ValueError("Fresh reload reused original compiled-model objects")
                test_cls.model = fresh
                case = test_cls("test_numerical_parity")
                case.test_numerical_parity()
                event({"phase": "fresh_reload_original_numerical_parity", "passed": True})
        finally:
            original_teardown()

    with ExitStack() as stack:
        if diagnostics is not None:
            from chronos import Chronos2Model

            cpu_forward = Chronos2Model.forward
            product_forward = RBLNChronos2Model.forward

            @wraps(cpu_forward)
            def cpu_capture(model, *args, **kwargs):
                inputs = {**kwargs, **({"context": args[0]} if args else {})}
                output = cpu_forward(model, *args, **kwargs)
                diagnostics.reference(model, inputs, output)
                return output

            @wraps(product_forward)
            def product_capture(model, *args, **kwargs):
                inputs = {**kwargs, **({"context": args[0]} if args else {})}
                with diagnostics.product_request(inputs):
                    output = product_forward(model, *args, **kwargs)
                    diagnostics.product_output(output)
                    return output

            stack.enter_context(patch.object(Chronos2Model, "forward", cpu_capture))
            stack.enter_context(patch.object(RBLNChronos2Model, "forward", product_capture))
        for name in ("_export", "compile"):
            stack.enter_context(patch.object(RBLNChronos2Model, name, classmethod(block_export)))
        stack.enter_context(
            patch.object(
                RBLNChronos2Model,
                "_create_runtimes",
                classmethod(
                    runtime_adapter(manifest, event, observer=None if diagnostics is None else diagnostics.stage)
                ),
            )
        )
        stack.enter_context(patch.object(cls, "setUpClass", classmethod(setup)))
        stack.enter_context(patch.object(cls, "tearDownClass", classmethod(teardown)))
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(cls)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    event(
        {
            "phase": "original_suite",
            "methods": methods,
            "tests_run": result.testsRun,
            "skipped": [(str(test), reason) for test, reason in result.skipped],
            "failures": len(result.failures),
            "errors": len(result.errors),
            "passed": result.wasSuccessful(),
        }
    )
    if result.testsRun != 9 or len(result.skipped) != 1 or "RBLN_AUTO_CLASS is None" not in result.skipped[0][1]:
        raise RuntimeError("Expected all nine original methods and only the existing automap skip")
    if not result.wasSuccessful():
        raise RuntimeError("Original native B512 suite failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime-library-sha256")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--capture-mask-diagnostics", action="store_true")
    args = parser.parse_args()
    args.artifacts = args.artifacts.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("SAVE_ARTIFACTS_PATH", "REUSE_ARTIFACTS_PATH"):
        if name in os.environ:
            raise ValueError("Artifact compatibility mode would skip native parity")
    manifest = validate_frozen(args.artifacts, args.manifest_sha256, manifest_name=args.manifest_name)
    if args.verify_only:
        print("FROZEN_VERIFY_ONLY_PASS; no runtime or NPU access")
        return
    if not os.environ.get("BUILDKITE_BUILD_ID"):
        raise RuntimeError("Native runtime execution requires a scheduler BUILDKITE_BUILD_ID")
    if not args.runtime_library_sha256:
        raise ValueError("Pin the installed SDK329 runtime library SHA256 before execution")
    provenance = verify_runtime_library(args.runtime_library_sha256)
    import rebel

    validate_frozen(args.artifacts, args.manifest_sha256, rebel.RBLNCompiledModel.inspect, args.manifest_name)
    events = args.out.resolve() / "events.jsonl"

    def event(value):
        with events.open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print("CHRONOS_FROZEN " + json.dumps(value), flush=True)

    event({"phase": "pre_runtime_verified", "runtime": provenance, "source_manifest_sha256": args.manifest_sha256})
    os.environ["OPTIMUM_RBLN_TEST_LEVEL"] = "full"
    os.environ["RBLN_TEST_NPU"] = "RBLN-CA25"
    os.chdir(args.out)
    diagnostics = None
    if args.capture_mask_diagnostics:
        from scripts.diagnostics.chronos2_mask_diagnostics import MaskDiagnostics

        diagnostics = MaskDiagnostics(args.out / "mask-diagnostics", event)
    run_suite(args.artifacts, manifest, args.out.resolve(), event, diagnostics=diagnostics)


if __name__ == "__main__":
    main()
