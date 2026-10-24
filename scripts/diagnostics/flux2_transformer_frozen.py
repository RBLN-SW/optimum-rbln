"""Scheduled real-weight native FLUX2 transformer diagnostic; full pipeline remains separate."""

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path


SDK = "0.12.1.dev333+gf66b88a7"
MANIFEST_SHA = "b569c12ed6b966c66be275a706ee59a0c6ea9c8adb48ed1203458eef1bf279e1"
BINARY_SHA = "860ea03c528db8d131557466762d7080bf1d6d2eafb6a899e79d7ecfcdb37829"
WRAPPER_SHA = "06b472d9368c86a66d3f32b2b729a7910e917d1244d716bdb20368bca3c0fa79"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
FILES = {"artifact/compiled_model.rbln", "artifact/config.json", "artifact/rbln_config.json", "golden.pt"}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bundle(directory):
    directory = Path(directory)
    if sha256(directory / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Frozen manifest differs from the reviewed native model")
    manifest = json.loads((directory / "manifest.json").read_text())
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if set(manifest["files"]) != FILES or actual != FILES | {"manifest.json"}:
        raise ValueError("Frozen bundle must contain exactly its reviewed files")
    for name, receipt in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Frozen files must be ordinary files inside the bundle")
        if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
            raise ValueError(f"Frozen checksum/size differs: {name}")
    expected = {
        "sdk": SDK,
        "revision": REVISION,
        "component": "transformer",
        "parameters": 3875544576,
        "target": "RBLN-CR13",
        "batch": 1,
        "image_size": [1024, 1024],
        "prompt_length": 512,
        "reference_image_count": 0,
        "teacher_forced_steps": 4,
        "recorded_compile_wrapper_sha256": WRAPPER_SHA,
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen model source, native geometry or target differs")
    if manifest["files"]["artifact/compiled_model.rbln"]["sha256"] != BINARY_SHA:
        raise ValueError("Unexpected trained compiled model")
    return manifest


def metrics(actual, expected):
    import torch

    a, b = actual.double().flatten(), expected.double().flatten()
    if not bool(torch.isfinite(b).all()) or b.square().mean() == 0 or b.std() == 0:
        raise ValueError("Native reference must be finite and nonconstant")
    finite = bool(torch.isfinite(a).all())
    if not finite:
        return {"finite": False, "pcc": None, "nrmse": None, "passed": False}
    centered_a, centered_b = a - a.mean(), b - b.mean()
    denominator = centered_a.norm() * centered_b.norm()
    pcc = (centered_a @ centered_b / denominator).item() if denominator > 0 else None
    nrmse = ((a - b).square().mean().sqrt() / b.square().mean().sqrt()).item()
    return {"finite": finite, "pcc": pcc, "nrmse": nrmse, "passed": pcc is not None and pcc >= 0.99 and nrmse <= 0.05}


def require_scheduled_execution():
    if not os.environ.get("BUILDKITE_BUILD_ID") or os.environ.get("RBLN_TEST_NPU") != "RBLN-CR13":
        raise ValueError("NPU execution requires the dedicated scheduled CR13 job")
    if os.environ.get("SAVE_ARTIFACTS_PATH") or os.environ.get("REUSE_ARTIFACTS_PATH"):
        raise ValueError("Artifact compatibility mode cannot establish NPU numerical acceptance")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("debug-flux2-transformer"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {"phase": "preflight", "component_numerical_passed": False, "full_pipeline_npu_acceptance": False}
    code = 1
    try:
        if args.execute:
            require_scheduled_execution()
        manifest = verify_bundle(args.artifacts)
        result["manifest_sha256"] = MANIFEST_SHA
        if args.verify_only:
            result["phase"] = "verified_only"
            code = 0
        else:
            import rebel
            import torch
            from tvm._ffi import base as ffi_base

            from optimum.rbln import RBLNFlux2Transformer2DModel

            if importlib.metadata.version("rebel-compiler") != SDK:
                raise ValueError("Wrong SDK for the frozen component")
            library = Path(ffi_base._LIB._name).resolve()
            if sha256(library) != manifest["compiler_dso_sha256"]:
                raise ValueError("Loaded compiler/runtime DSO differs from the reviewed SDK333 wheel")
            module = importlib.import_module(RBLNFlux2Transformer2DModel.__module__)
            if sha256(Path(module.__file__)) != WRAPPER_SHA:
                raise ValueError("Imported wrapper source differs from the compiled native wrapper")
            metadata = rebel.RBLNCompiledModel.inspect(args.artifacts / "artifact/compiled_model.rbln")
            if metadata["inputs"] != manifest["inputs"] or metadata["outputs"] != manifest["outputs"]:
                raise ValueError("Compiled native interfaces differ from the reviewed manifest")
            if metadata["npu"] != "RBLN-CR13" or metadata["num_devices"] != 1:
                raise ValueError("Compiled model targets a different device")
            if rebel.get_npu_name(0) != "RBLN-CR13":
                raise ValueError("Scheduled physical device does not match the compile target")
            result.update(sdk=SDK, compiler_dso_sha256=manifest["compiler_dso_sha256"], metadata=metadata)
            golden = torch.load(args.artifacts / "golden.pt", weights_only=True)
            cases = golden["denoiser"]
            if len(cases) != 4:
                raise ValueError("Expected all four native denoising-step references")
            model = RBLNFlux2Transformer2DModel.from_pretrained(
                args.artifacts / "artifact", rbln_create_runtimes=True, rbln_device=0
            )
            result["phase"] = "execute"
            actuals, rows = [], []
            result["rows"] = rows
            with torch.inference_mode():
                for index, case in enumerate(cases):
                    inputs = {k: v for k, v in case.items() if k != "noise"}
                    before = time.monotonic()
                    actual = model(**inputs, return_dict=False)[0].detach().clone()
                    actuals.append(actual)
                    torch.save({"actual": actuals}, args.output / "actual.pt")
                    row = {"step": index, "seconds": time.monotonic() - before, **metrics(actual, case["noise"])}
                    rows.append(row)
                    repeated = model(**inputs, return_dict=True).sample
                    row["repeat_and_return_dict_exact"] = torch.equal(repeated, actual)
                    print(json.dumps(row), flush=True)
                first = {k: v for k, v in cases[0].items() if k != "noise"}
                result["repeat_and_return_exact"] = all(
                    r["repeat_and_return_dict_exact"] for r in rows
                ) and torch.equal(model(**first, return_dict=False)[0], actuals[0])
                reloaded = RBLNFlux2Transformer2DModel.from_pretrained(
                    args.artifacts / "artifact", rbln_create_runtimes=True, rbln_device=0
                )
                if reloaded is model or reloaded.model[0] is model.model[0]:
                    raise ValueError("Reload must create an independent runtime wrapper")
                result["reload_exact"] = torch.equal(reloaded(**first, return_dict=False)[0], actuals[0])
            result.update(
                phase="finished",
                component_numerical_passed=all(r["passed"] for r in rows)
                and result["repeat_and_return_exact"]
                and result["reload_exact"],
            )
            code = 0 if result["component_numerical_passed"] else 1
            del reloaded, model
            gc.collect()
    except BaseException:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        result.update(exit_code=code, seconds=time.monotonic() - started)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        suite = ET.Element("testsuite", name="flux2_transformer_native1024", tests="1", failures=str(int(code != 0)))
        case = ET.SubElement(suite, "testcase", name=result["phase"])
        if code:
            ET.SubElement(case, "failure").text = result.get("error", "Native component numerical threshold failed")
        ET.ElementTree(suite).write(args.output / "junit.xml", encoding="unicode")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
