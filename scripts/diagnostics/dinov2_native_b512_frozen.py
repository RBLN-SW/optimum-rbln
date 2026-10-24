"""Run unchanged native B512 tests against checksum-verified frozen CA25 artifacts."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
import traceback
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


COMPILER = "66a3078e939b6af2e6e7f745304bc5e599f0a77f"
OPTIMUM = "e162e1ae3a38e1a6d46418f6a57e01235992f4da"
SDK = "0.12.1.dev329+g66a3078e"
STAGES = ["embedding", "encoder_0", "encoder_6", "tail"]
CLASSES = {
    "TestDinov2ModelNativeSizeBatch512": "encoder",
    "TestDinov2ForImageClassificationNativeSizeBatch512": "classifier",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(root, expected_compiler=COMPILER, expected_optimum=OPTIMUM, inspect_model=None):
    """Verify every frozen file and saved full-batch interface without creating runtimes."""
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["source_compiler"] != expected_compiler or manifest["source_optimum"] != expected_optimum:
        raise ValueError("Frozen artifact source heads do not match the required sources")
    files = manifest["files"]
    actual = {str(p.relative_to(root)) for kind in CLASSES.values() for p in (root / kind).rglob("*") if p.is_file()}
    if set(files) != actual:
        raise ValueError("Manifest must list every encoder/classifier file exactly")
    for name, receipt in files.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe artifact path: {name}")
        if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
            raise ValueError(f"Artifact checksum/size mismatch: {name}")
    if inspect_model is None:
        import rebel

        inspect_model = rebel.RBLNCompiledModel.inspect
    metadata = {}
    for kind in CLASSES.values():
        directory = root / kind
        config = json.loads((directory / "rbln_config.json").read_text())
        model_config = json.loads((directory / "config.json").read_text())
        if config["batch_size"] != 512 or config["image_size"] not in (518, [518, 518]):
            raise ValueError(f"Wrong native geometry: {kind}")
        if config["encoder_chunk_size"] != 6 or model_config["num_hidden_layers"] != 12:
            raise ValueError(f"Wrong encoder segmentation: {kind}")
        configs = config["_compile_cfgs"]
        if [cfg["compiled_model_name"] for cfg in configs] != STAGES:
            raise ValueError(f"Wrong compiled stage names/order: {kind}")
        for cfg in configs:
            inputs = {name: (shape, dtype) for name, shape, dtype in cfg["input_info"]}
            if cfg["npu"] != "RBLN-CA25" or cfg["num_devices"] != 1:
                raise ValueError(f"Wrong target: {kind}")
            if inputs["activation"] != ([512, 1370, 768], "float32"):
                raise ValueError(f"Wrong shared activation: {kind}")
        if configs[0]["input_info"][0] != ["pixel_values", [512, 3, 518, 518], "float32"]:
            raise ValueError(f"Wrong pixel input: {kind}")
        rows = [inspect_model(directory / f"{name}.rbln") for name in STAGES]
        if len({row["uuid"] for row in rows}) != 1 or any(row["npu"] != "RBLN-CA25" for row in rows):
            raise ValueError(f"Stages must share one CA25 compile session: {kind}")
        metadata[kind] = rows
    return {"manifest": manifest, "metadata": metadata}


class Recorder:
    def __init__(self, directory):
        self.directory = directory
        self.test = None
        self.loads = 0
        self.failed = []
        self.skipped = []
        self.collected = 0

    def event(self, **row):
        row.update(test=self.test, time_ns=time.time_ns())
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()

    def pytest_collection_modifyitems(self, items):
        self.collected = len(items)
        if self.collected != 18 or {item.cls.__name__ for item in items} != set(CLASSES):
            raise ValueError("Expected all nine original methods from each native B512 class")

    def pytest_runtest_logstart(self, nodeid, location):
        self.test = nodeid
        self.event(phase="test_start")

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.failed.append(f"{report.nodeid}:{report.when}")
        if report.skipped:
            self.skipped.append(f"{report.nodeid}:{report.when}")
        self.event(phase="test_report", outcome=report.outcome, when=report.when)

    def stage(self, runtime, name, load):
        calls = 0

        def timed(*args, **kwargs):
            nonlocal calls
            calls += 1
            row = {"stage": name, "load": load, "call": calls, "temperature": "cold" if calls == 1 else "warm"}
            self.event(phase="stage_start", **row)
            start = time.perf_counter()
            error = None
            try:
                return runtime(*args, **kwargs)
            except BaseException as exc:
                error = repr(exc)
                raise
            finally:
                self.event(phase="stage_end", seconds=time.perf_counter() - start, error=error, **row)

        return timed


def frozen_setup(source):
    def setup(cls):
        destination = Path(cls.get_rbln_local_dir()).resolve()
        if destination == source.resolve() or source.resolve() in destination.parents:
            raise ValueError("Tests must copy frozen artifacts into an independent local directory")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        cls.model = cls.RBLN_CLASS.from_pretrained(destination, rbln_device=0)

    return classmethod(setup)


def run_tests(root, recorder):
    import pytest

    from optimum.rbln.transformers.models.dinov2.modeling_dinov2 import Dinov2Mixin
    from tests import test_transformers

    original = Dinov2Mixin.__dict__["_create_runtimes"].__func__

    def create(cls, compiled_models, rbln_config):
        recorder.loads += 1
        load = recorder.loads
        recorder.event(phase="runtime_create_start", load=load, model=cls.__name__)
        try:
            chains = original(cls, compiled_models, rbln_config)
            for chain in chains:
                chain.runtimes = [
                    recorder.stage(runtime, cfg.compiled_model_name, load)
                    for runtime, cfg in zip(chain.runtimes, rbln_config.compile_cfgs, strict=True)
                ]
            return chains
        finally:
            recorder.event(phase="runtime_create_end", load=load, model=cls.__name__)

    with ExitStack() as stack:
        stack.enter_context(patch.object(Dinov2Mixin, "_create_runtimes", classmethod(create)))
        stack.enter_context(patch.dict(os.environ, {"OPTIMUM_RBLN_TEST_LEVEL": "full", "PYTEST_ADDOPTS": ""}))
        for name, kind in CLASSES.items():
            stack.enter_context(
                patch.object(getattr(test_transformers, name), "setUpClass", frozen_setup(root / kind))
            )
        nodes = [f"tests/test_transformers.py::{name}" for name in CLASSES]
        return int(
            pytest.main(
                [
                    *nodes,
                    "-vv",
                    "-s",
                    "--durations=0",
                    "-o",
                    "addopts=",
                    f"--junitxml={recorder.directory / 'junit.xml'}",
                ],
                plugins=[recorder],
            )
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("debug-dinov2-native-b512-frozen"))
    parser.add_argument("--expected-compiler", default=COMPILER)
    parser.add_argument("--expected-optimum", default=OPTIMUM)
    parser.add_argument(
        "--verify-only", action="store_true", help="Check hashes/config/metadata; never create a runtime"
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("events.jsonl", "junit.xml", "result.json"):
        (args.output / name).unlink(missing_ok=True)
    recorder = Recorder(args.output.resolve())
    result = {"phase": "verify", "accepted": False, "runtime_executed": False}
    code = 1
    try:
        if os.environ.get("REUSE_ARTIFACTS_PATH") or os.environ.get("SAVE_ARTIFACTS_PATH"):
            raise ValueError("Artifact compatibility environment would skip numerical acceptance")
        result["verification"] = verify_artifacts(
            args.artifacts.resolve(), args.expected_compiler, args.expected_optimum
        )
        if args.verify_only:
            result["phase"] = "verified_only"
            code = 0
        else:
            result["sdk"] = importlib.metadata.version("rebel-compiler")
            if result["sdk"] != SDK:
                raise ValueError(f"Expected SDK {SDK}, got {result['sdk']}")
            result.update(phase="tests", runtime_executed=True)
            code = run_tests(args.artifacts.resolve(), recorder)
            result.update(collected=recorder.collected, failed=recorder.failed, skipped=recorder.skipped)
            if recorder.skipped or recorder.collected != 18:
                code = 1
            result.update(phase="finished", accepted=code == 0)
    except BaseException:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        result["exit_code"] = code
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        junit = args.output / "junit.xml"
        if not junit.exists():
            suite = ET.Element("testsuite", name="dinov2_frozen_preflight", tests="1", failures=str(int(code != 0)))
            case = ET.SubElement(suite, "testcase", name=result["phase"])
            if code:
                ET.SubElement(case, "failure").text = result.get("error", "Preflight failed")
            ET.ElementTree(suite).write(junit, encoding="unicode")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
