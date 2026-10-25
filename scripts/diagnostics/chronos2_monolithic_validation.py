"""Validate the original unsplit Chronos B512 model with saved compiler-fix bytes."""

import argparse
import json
import os
import shutil
from pathlib import Path

from scripts.diagnostics import chronos2_frozen_validation as original
from scripts.diagnostics.chronos2_candidate_validation import BASELINE, require_candidate_source


def validate_bundle(root, digest, inspect_model=None):
    root = Path(root)
    if original.sha256(root / "manifest.json") != digest:
        raise ValueError("Monolithic manifest SHA mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    require_candidate_source(manifest)
    if (
        manifest["kind"] != "original-unsplit-Chronos-B512-compiler-fix"
        or manifest["full_model_acceptance"] is not False
    ):
        raise ValueError("Invalid unexecuted monolithic candidate scope")
    for name, row in manifest["files"].items():
        path = root / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ValueError("Frozen monolithic file escaped its root")
        if path.stat().st_size != row["bytes"] or original.sha256(path) != row["sha256"]:
            raise ValueError("Frozen monolithic file bytes changed: " + name)
    if {p.name for p in root.iterdir()} != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Missing or extra monolithic frozen files")
    if original.sha256(root / "baseline-manifest.json") != BASELINE:
        raise ValueError("Original baseline changed")
    baseline = json.loads((root / "baseline-manifest.json").read_text())
    for name in ["config.json", "original_rbln_config.json", "original-test-generate-inputs.pt"]:
        if manifest["files"][name]["sha256"] != baseline["files"][name]["sha256"]:
            raise ValueError("Original model config/input bytes changed")
    if original.sha256(root / "rbln_config.json") != baseline["files"]["original_rbln_config.json"]["sha256"]:
        raise ValueError("Product config must retain its original single compile interface")
    config = json.loads((root / "rbln_config.json").read_text())
    if (config["batch_size"], config["context_length"], config["prediction_length"]) != (512, 8192, 1024):
        raise ValueError("Original native maximum geometry changed")
    if manifest["compile_configs"] != config["_compile_cfgs"] or len(manifest["stages"]) != 1:
        raise ValueError("The original monolithic model must retain one stage")
    if manifest["stages"][0]["name"] != "compiled_model" or manifest["stages"][0]["file"] != "compiled_model.rbln":
        raise ValueError("Original compiled-model name changed")
    report = json.loads((root / "compile-result.json").read_text())
    require_candidate_source(report)
    if report["phase"] != "compiled" or not report["all_checkpoint_weight_tensors_exact"] or report["guard_calls"]:
        raise ValueError("Original checkpoint/compile provenance missing")
    if len(report["checkpoint_weights"]) != 170 or len(report["stage_weights"]) != 170:
        raise ValueError("All original trained tensors must be retained")
    if {w["native_name"] for w in report["stage_weights"]} != set(report["checkpoint_weights"]):
        raise ValueError("Native trained tensor ownership changed")
    if any(w["sha256"] != report["checkpoint_weights"][w["native_name"]]["sha256"] for w in report["stage_weights"]):
        raise ValueError("Original trained tensor values changed")
    if report["placement"]["learned_arithmetic_host"] or report["placement"]["device_functions"] != 1:
        raise ValueError("Learned arithmetic must execute in the original device graph")
    metadata = report["metadata"]
    if inspect_model is not None and inspect_model(root / "compiled_model.rbln") != metadata:
        raise ValueError("Actual compiled artifact metadata differs from the recorded interface")
    if report["sha256"] != manifest["files"]["compiled_model.rbln"]["sha256"]:
        raise ValueError("Compiled artifact/report checksum binding changed")
    if metadata["npu"] != "RBLN-CA25" or metadata["num_devices"] != 1:
        raise ValueError("Expected exactly one native CA25 target")
    if [[row["name"], row["shape"], row["dtype"]] for row in metadata["inputs"]] != config["_compile_cfgs"][0][
        "input_info"
    ]:
        raise ValueError("Original native input interface changed")
    if len(metadata["outputs"]) != 2 or any(
        row["shape"] != [512, 21, 1024] or row["dtype"] != "float32" for row in metadata["outputs"]
    ):
        raise ValueError("Original FP32 forecast/residual tuple changed")
    for name, digest in manifest["source_sha256"].items():
        path = Path(__file__).parents[2] / name
        if original.sha256(path) != digest:
            raise ValueError("Reviewed source binding changed: " + name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stage-to", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.bundle = args.bundle.resolve()
    manifest = validate_bundle(args.bundle, args.manifest_sha256)
    if args.stage_to is not None:
        args.stage_to.parent.mkdir(parents=True, exist_ok=True)
        if args.stage_to.exists():
            validate_bundle(args.stage_to, args.manifest_sha256)
        else:
            temporary = args.stage_to.with_name(args.stage_to.name + f".partial-{os.getpid()}")
            if temporary.exists():
                raise ValueError("Retaining an existing partial staging directory")
            shutil.copytree(args.bundle, temporary)
            validate_bundle(temporary, args.manifest_sha256)
            temporary.rename(args.stage_to)
        print("MONOLITHIC_STAGED_VERIFIED; no NPU APIs", flush=True)
        return
    if args.verify_only:
        print("MONOLITHIC_VERIFY_ONLY_PASS; no compiler/NPU imports", flush=True)
        return
    if not os.environ.get("BUILDKITE_JOB_ID") or os.environ.get("RBLN_TEST_NPU") != "RBLN-CA25":
        raise ValueError("Execution requires the scheduler-owned CA25 job")
    if any(
        name in os.environ
        for name in ["RBLN_DUMMY_DEVICE", "TVM_LIBRARY_PATH", "SAVE_ARTIFACTS_PATH", "REUSE_ARTIFACTS_PATH"]
    ):
        raise ValueError("Dummy/override/artifact compatibility modes are forbidden")
    if args.out is None:
        raise ValueError("Execution requires an output directory")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    runtime = original.verify_runtime_library("4e1bc153f001ae5a8694478052c7df4b05d8b5dbc09c6a27c06153e6decc7a9e")
    import rebel

    from scripts.diagnostics.chronos2_prediction_audit import PredictionAudit

    validate_bundle(args.bundle, args.manifest_sha256, rebel.RBLNCompiledModel.inspect)
    report = json.loads((args.bundle / "compile-result.json").read_text())
    audit = PredictionAudit(args.out / "prediction-audit", report["checkpoint_weights"])
    events = args.out / "events.jsonl"

    def event(value):
        with events.open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print("CHRONOS_MONOLITHIC " + json.dumps(value), flush=True)

    event({"phase": "pre_runtime_verified", "runtime": runtime, "original_native_runtime_factory": True})
    os.environ["OPTIMUM_RBLN_TEST_LEVEL"] = "full"
    os.chdir(args.out)
    original_passed = False
    try:
        original.run_suite(args.bundle, manifest, args.out, event, diagnostics=audit, sequential=False)
        original_passed = True
    finally:
        prediction_report = audit.finish()
        event(
            {
                "phase": "prediction_audit",
                "matched_calls": prediction_report["matched_calls"],
                "reference_calls": len(prediction_report["reference_calls"]),
                "unique_reference_inputs": prediction_report["unique_reference_inputs"],
                "all_series_passed": prediction_report["all_series_passed"],
                "identical_input_outputs_exact": prediction_report["identical_input_outputs_exact"],
            }
        )
    if (
        not original_passed
        or not prediction_report["all_series_passed"]
        or not prediction_report["identical_input_outputs_exact"]
    ):
        raise RuntimeError("Original monolithic/per-series numerical acceptance failed")
    if (
        prediction_report["matched_calls"] != 22
        or len(prediction_report["reference_calls"]) != 10
        or prediction_report["unique_reference_inputs"] != 5
    ):
        raise RuntimeError("Original five scenarios/repeats/reload were not all observed")


if __name__ == "__main__":
    main()
