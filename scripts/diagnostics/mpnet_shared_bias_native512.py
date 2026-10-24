"""Scheduler-only proof of the reviewed shared-bias candidate at native512 B1/B2/B4."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


BASE = "e5acc2960ec48eb3bcf122ad53428ff878a775f2"
SDK = "0.12.1.dev163+g0c6d4a5e.prod"
SDK_LIBRARY_SHA = "ff6af17e0ff975ac22100e92da58f74bb61ba06999c94aceb4c829c6654f96ed"
CLASSES = ("TestMPNetModelNativeMaxBatch1", "TestMPNetModelNativeMaxBatch2", "TestMPNetModelNativeMaxBatch4")
METHODS = (
    "test_automap",
    "test_embedding_parity",
    "test_generate",
    "test_get_rbln_config_class",
    "test_infer_framework",
    "test_model_save_dir",
    "test_model_save_dir_load",
    "test_save_artifacts",
    "test_save_load",
)
SOURCE_SHA = {
    "src/optimum/rbln/transformers/models/mpnet/mpnet_architecture.py": "c98290805e7ef3b051c5dcde2d5569f6b518fe867339a2e6091244998301830c",
    "tests/test_transformers.py": "12ed1e4b5ba31d0ffde0c1362dd9957c260e9e1665a97288e2134aac45749c67",
    "tests/test_base.py": "c1ea049b86f9a125714e19fbee0d7974c1ad26e260ad3fdf7d8549c679d63b4d",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def guard_environment(verify_only):
    for name in ("SAVE_ARTIFACTS_PATH", "REUSE_ARTIFACTS_PATH"):
        if name in os.environ:
            raise ValueError(f"{name} would alter the original native acceptance path")
    if not verify_only and not os.environ.get("BUILDKITE_BUILD_ID"):
        raise RuntimeError("NPU compile/runtime execution requires scheduler BUILDKITE_BUILD_ID")
    if not verify_only and os.environ.get("RBLN_DUMMY_DEVICE"):
        raise ValueError("Native acceptance cannot run on the dummy device")


def verify_sources(root):
    for name, expected in SOURCE_SHA.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Reviewed candidate or original acceptance source changed: {name}")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=root, check=True)
    changed_product = subprocess.check_output(["git", "diff", "--name-only", BASE, "--", "src"], cwd=root, text=True)
    if changed_product.splitlines() != ["src/optimum/rbln/transformers/models/mpnet/mpnet_architecture.py"]:
        raise ValueError("Only the reviewed architecture candidate may change product code")
    return {
        "product_base": BASE,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": SOURCE_SHA,
    }


def select_original_classes():
    from optimum.rbln.transformers.models.mpnet import mpnet_architecture
    from tests import test_base, test_transformers

    for module, name in ((test_base, "tests/test_base.py"), (test_transformers, "tests/test_transformers.py")):
        if sha256(Path(module.__file__)) != SOURCE_SHA[name]:
            raise ValueError(f"Original acceptance import resolves to different code: {name}")

    if (
        sha256(Path(mpnet_architecture.__file__))
        != SOURCE_SHA["src/optimum/rbln/transformers/models/mpnet/mpnet_architecture.py"]
    ):
        raise ValueError("Candidate import resolves to a different checkout")
    classes = [getattr(test_transformers, name) for name in CLASSES]
    selection = []
    for cls, batch in zip(classes, (1, 2, 4), strict=True):
        names = unittest.defaultTestLoader.getTestCaseNames(cls)
        if tuple(names) != METHODS:
            raise ValueError(f"Original nine-method selection changed: {cls.__name__}: {names}")
        if batch != cls.BATCH_SIZE or cls.RBLN_CLASS_KWARGS["rbln_max_seq_len"] != 512:
            raise ValueError("Expected native512 B1/B2/B4 only")
        if cls.RBLN_CLASS_KWARGS["rbln_batch_size"] != batch:
            raise ValueError("Configured batch does not match test class")
        if cls.HF_MODEL_ID != "sentence-transformers/all-mpnet-base-v2" or cls.HF_CONFIG_KWARGS != {
            "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130"
        }:
            raise ValueError("Original native checkpoint/revision changed")
        if any(list(value.shape) != [batch, 512] for value in cls.GENERATION_KWARGS.values()):
            raise ValueError("Original native generate geometry changed")
        selection.extend(f"{cls.__module__}.{cls.__name__}.{name}" for name in names)
    if len(selection) != 27:
        raise ValueError("Expected exactly27 original methods")
    return classes, selection


def verify_sdk():
    import rebel

    version = importlib.metadata.version("rebel-compiler")
    if version != SDK:
        raise ValueError(f"Expected {SDK}, got {version}")
    mapped = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "/librbln.so" in line}
    if len(mapped) != 1 or sha256(next(iter(mapped))) != SDK_LIBRARY_SHA:
        raise ValueError("Expected sole published SDK163.prod DSO")
    distribution = importlib.metadata.distribution("rebel-compiler")
    if not Path(rebel.__file__).samefile(distribution.locate_file("rebel/__init__.py")):
        raise ValueError("SDK Python import does not match its installed wheel")
    return {
        "version": version,
        "library": next(iter(mapped)),
        "library_sha256": SDK_LIBRARY_SHA,
        "rebel_module": rebel.__file__,
    }


def fresh_reload_original_parity(cls, work, event, context_factory=None):
    """Release the original runtimes, load new objects, and call the unchanged oracle."""
    if context_factory is None:
        from optimum.rbln.utils.runtime_utils import ContextRblnConfig

        context_factory = ContextRblnConfig
    with tempfile.TemporaryDirectory(dir=work) as temporary:
        cls.model.save_pretrained(temporary)
        prior = cls.model.compiled_models
        saved = {path.name: sha256(path) for path in Path(temporary).glob("*.rbln")}
        if not saved:
            raise ValueError("Fresh reload has no saved compiled artifacts")
        del cls.model
        gc.collect()
        try:
            with context_factory(device=cls.DEVICE):
                cls.model = cls.RBLN_CLASS.from_pretrained(temporary, export=False, **cls.HF_CONFIG_KWARGS)
            if any(old is new for old, new in zip(prior, cls.model.compiled_models, strict=True)):
                raise ValueError("Fresh reload reused original compiled-model objects")
            case = cls("test_embedding_parity")
            case.test_embedding_parity()
            event(
                {
                    "phase": "fresh_reload_original_embedding_parity",
                    "class": cls.__name__,
                    "passed": True,
                    "saved_compiled_sha256": saved,
                    "fresh_compiled_object_ids": [id(x) for x in cls.model.compiled_models],
                }
            )
        finally:
            if "model" in cls.__dict__:
                del cls.model
            gc.collect()


def run_original_suite(classes, work, event):
    def teardown_for(cls):
        original = cls.tearDownClass

        def teardown(test_cls):
            try:
                fresh_reload_original_parity(test_cls, work, event)
            finally:
                original()

        return classmethod(teardown)

    with ExitStack() as stack:
        for cls in classes:
            stack.enter_context(patch.object(cls, "tearDownClass", teardown_for(cls)))
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in classes)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    event(
        {
            "phase": "original_suite_result",
            "tests_run": result.testsRun,
            "skipped": [(str(test), reason) for test, reason in result.skipped],
            "failures": len(result.failures),
            "errors": len(result.errors),
            "passed": result.wasSuccessful(),
        }
    )
    if result.testsRun != 27 or result.skipped or not result.wasSuccessful():
        raise RuntimeError("Original27-method native512 MPNet candidate suite failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    guard_environment(args.verify_only)
    root = Path(__file__).resolve().parents[2]
    provenance = verify_sources(root)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["OPTIMUM_RBLN_TEST_LEVEL"] = "full"
    os.environ["RBLN_TEST_NPU"] = "RBLN-CR13"
    classes, selection = select_original_classes()
    provenance.update(
        {
            "selection": selection,
            "original_methods": 27,
            "extra_fresh_reload_parity_invocations": 3,
            "batches": [1, 2, 4],
            "max_seq_len": 512,
            "note": "Small native matrix validates shared bias against actual row masks; it does not establish B2048 capacity.",
        }
    )
    (args.out / "selection.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.verify_only:
        print("MPNET_NATIVE512_VERIFY_ONLY_PASS:27 original methods; no compile/runtime/NPU")
        return
    provenance["runtime"] = verify_sdk()
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    events = args.out / "events.jsonl"

    def event(value):
        with events.open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print("MPNET_NATIVE512 " + json.dumps(value), flush=True)

    os.chdir(args.out)
    run_original_suite(classes, args.out, event)


if __name__ == "__main__":
    main()
