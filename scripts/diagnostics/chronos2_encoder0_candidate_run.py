"""Compare a compiler fix on the original Chronos inputs and trained blocks."""

import argparse
import gc
import importlib.metadata
import json
import os
import time
from pathlib import Path

from scripts.diagnostics.chronos2_encoder0_probe_run import file_sha, verify_bundle


def boundary_metrics(actual, expected):
    import torch

    actual, expected = actual.double().flatten(), expected.double().flatten()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Nonfinite boundary values")
    error = (actual - expected).square().mean().sqrt().item()
    rms = expected.square().mean().sqrt().item()
    left, right = actual - actual.mean(), expected - expected.mean()
    denominator = (left.norm() * right.norm()).item()
    nrmse = error / rms if rms else None
    pcc = torch.dot(left, right).item() / denominator if denominator else None
    passed = nrmse is not None and nrmse <= 0.05 and pcc is not None and pcc >= 0.99
    if not rms:
        passed = torch.equal(actual, expected)
    return {"nrmse": nrmse, "pcc": pcc, "max_abs": (actual - expected).abs().max().item(), "passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = verify_bundle(args.bundle, args.manifest_sha256)
    if manifest["kind"] != "compiler-fix-candidate-original-Chronos-encoder0":
        raise ValueError("This runner requires an explicit compiler-fix candidate bundle")
    if file_sha(Path(__file__)) != manifest["source_sha256"]["candidate_runner"]:
        raise ValueError("Candidate runner source mismatch")
    if not os.environ.get("BUILDKITE_JOB_ID") or os.environ.get("RBLN_TEST_NPU") != "RBLN-CA25":
        raise ValueError("Runtime execution requires the scheduler-owned CA25 job")
    if os.environ.get("RBLN_DUMMY_DEVICE") or os.environ.get("TVM_LIBRARY_PATH"):
        raise ValueError("Dummy devices and compiler-library overrides are forbidden")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {
        "phase": "preflight",
        "full_model_acceptance": False,
        "original_instrumentation_identity_claim": False,
        "scope": "Compiler fix on original full512 first3 trained blocks; not whole-model acceptance",
        "manifest_sha256": args.manifest_sha256,
    }

    def event(phase):
        result.update(phase=phase, seconds=time.monotonic() - started)
        (args.out / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
        print(phase, flush=True)

    try:
        import rebel
        import torch
        from tvm._ffi import base as ffi_base

        from scripts.diagnostics import chronos2_encoder0_probe
        from scripts.diagnostics.chronos2_encoder0_probe import OUTPUT_NAMES, tensor_sha256

        if file_sha(chronos2_encoder0_probe.__file__) != manifest["source_sha256"]["probe"]:
            raise ValueError("Trained probe source mismatch")
        if importlib.metadata.version("rebel-compiler") != manifest["runtime_SDK"]:
            raise ValueError("Runtime SDK mismatch")
        if file_sha(Path(ffi_base._LIB._name)) != manifest["runtime_library_sha256"]:
            raise ValueError("Actual runtime DSO mismatch")
        compile_result = json.loads((args.bundle / "compile-result.json").read_text())
        if not compile_result["all_weight_hashes_exact"] or compile_result["trained_parameter_tensors"] != 39:
            raise ValueError("Original 39 trained weight bindings missing")
        if len(compile_result["candidate_source_overlay"]) != 2:
            raise ValueError("Expected the two recorded candidate lowering functions")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        inputs = torch.load(args.bundle / "exact-ci819-inputs.pt", map_location="cpu", weights_only=True)
        receipt = json.loads((args.bundle / "original-inputs.json").read_text())["inputs"]
        names = ["hidden", "attention_mask", "group_time_mask"]
        for name in names:
            if list(inputs[name].shape) != receipt[name]["shape"] or str(inputs[name].dtype) != receipt[name]["dtype"]:
                raise ValueError("Original input interface changed")
            if tensor_sha256(inputs[name]) != receipt[name]["sha256"]:
                raise ValueError("Original input bytes changed")
        result["original_inputs_exact"] = True
        model = rebel.RBLNCompiledModel(args.bundle / "encoder0_boundaries.rbln")
        runtime = model.create_runtime(tensor_type="pt", device=0)

        def run():
            values = runtime(*(inputs[name] for name in names))
            if not isinstance(values, (tuple, list)) or len(values) != 5:
                raise ValueError("Expected five original full-sized boundaries")
            hashes = {}
            for name, value in zip(OUTPUT_NAMES, values, strict=True):
                if list(value.shape) != [512, 577, 768] or value.dtype != torch.float32:
                    raise ValueError("Original boundary interface changed")
                hashes[name] = tensor_sha256(value)
            return values, hashes

        event("first_candidate_execution")
        outputs, hashes = run()
        retained = {
            name: {"active": value[:3].clone(), "selected_padded": value[[3, 4, 511]].clone()}
            for name, value in zip(OUTPUT_NAMES, outputs, strict=True)
        }
        del outputs
        torch.save(retained, args.out / "selected-boundaries.pt")
        result["full_output_sha256"] = hashes
        result["historical_BAD_final_sha256"] = manifest["original_final_device_sha256"]
        event("ten_repeats")
        result["repeat_output_sha256"] = []
        for _ in range(10):
            outputs, repeated = run()
            del outputs
            result["repeat_output_sha256"].append(repeated)
            if repeated != hashes:
                raise ValueError("Candidate repeat differs")
        del runtime, model
        gc.collect()
        model = rebel.RBLNCompiledModel(args.bundle / "encoder0_boundaries.rbln")
        runtime = model.create_runtime(tensor_type="pt", device=0)
        event("fresh_artifact_reload")
        outputs, reloaded = run()
        del outputs
        result["reload_output_sha256"] = reloaded
        if reloaded != hashes:
            raise ValueError("Candidate artifact reload differs")
        result["postrun_input_sha256"] = {name: tensor_sha256(inputs[name]) for name in names}
        if any(result["postrun_input_sha256"][name] != receipt[name]["sha256"] for name in names):
            raise ValueError("Runtime changed original input bytes")
        references = torch.load(args.bundle / "cpu-boundary-references.pt", map_location="cpu", weights_only=True)
        result["boundaries"] = {
            name: {
                part: boundary_metrics(retained[name][part], references[name][part])
                for part in ["active", "selected_padded"]
            }
            for name in OUTPUT_NAMES
        }
        result["all_selected_boundary_criteria_passed"] = all(
            metrics["passed"] for parts in result["boundaries"].values() for metrics in parts.values()
        )
        event(
            "candidate_boundaries_passed"
            if result["all_selected_boundary_criteria_passed"]
            else "candidate_boundary_parity_failed"
        )
    except Exception as error:
        result.update(phase="failed", error_type=type(error).__name__, error=str(error))
    finally:
        result["seconds"] = time.monotonic() - started
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["phase"] != "candidate_boundaries_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
