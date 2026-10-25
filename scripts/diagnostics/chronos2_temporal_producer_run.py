"""Verify frozen inputs, then observe existing Chronos boundaries on scheduler CA25."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
import traceback
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(root, expected_manifest_sha):
    root = Path(root).resolve()
    file = root / "manifest.json"
    if file_sha(file) != expected_manifest_sha:
        raise ValueError("Manifest SHA mismatch; no runtime may be created")
    manifest = json.loads(file.read_text())
    expected = {
        "encoder0_producers.rbln",
        "cpu-producer-references.pt",
        "exact-ci819-inputs.pt",
        "cpu-boundary-references.pt",
        "original-inputs.json",
        "trained-parameters.json",
        "cpu-probe-receipt.json",
        "compile-result.json",
    }
    if set(manifest["files"]) != expected:
        raise ValueError("Unexpected frozen bundle membership")
    found = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if found != expected | {"manifest.json"}:
        raise ValueError("Missing or extra frozen files")
    for name, row in manifest["files"].items():
        path = (root / name).resolve()
        if root not in path.parents or (root / name).is_symlink() or not path.is_file():
            raise ValueError("Frozen file escaped its bundle")
        if path.stat().st_size != row["bytes"] or file_sha(path) != row["sha256"]:
            raise ValueError("Frozen file size/SHA mismatch: " + name)
    if manifest["full_model_acceptance"] is not False or manifest["target"] != "RBLN-CA25":
        raise ValueError("Invalid diagnostic scope")
    if manifest.get("observation_row_indices") != [0, 1, 2, 3, 4, 511]:
        raise ValueError("Producer observation row mapping changed")
    if manifest.get("model_input_geometry") != [512, 577, 768]:
        raise ValueError("Original model geometry changed")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = verify_bundle(args.bundle, args.manifest_sha256)
    if args.verify_only:
        print(
            json.dumps(
                {
                    "verified_files": len(manifest["files"]),
                    "NPU_API_calls": 0,
                    "runtime_created": False,
                    "full_model_acceptance": False,
                }
            ),
            flush=True,
        )
        return
    if not os.environ.get("BUILDKITE_JOB_ID") or os.environ.get("RBLN_TEST_NPU") != "RBLN-CA25":
        raise ValueError("Execution is restricted to the reserved scheduler CA25 job")
    if os.environ.get("RBLN_DUMMY_DEVICE") or os.environ.get("TVM_LIBRARY_PATH"):
        raise ValueError("Frozen execution forbids dummy device and compiler-library overrides")
    if args.out is None:
        raise ValueError("An output directory is required for execution")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {
        "phase": "runtime_preflight",
        "full_model_acceptance": False,
        "valid_for_original_encoder0_attribution": False,
        "source": "CI819 exact request0006",
        "model": "amazon/chronos-2",
        "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
        "full_geometry": [512, 577, 768],
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

        from scripts.diagnostics import chronos2_encoder0_probe, chronos2_temporal_producer_probe
        from scripts.diagnostics.chronos2_encoder0_probe import (
            ORIGINAL_FINAL_SHA256,
            require_original_final_identity,
            tensor_sha256,
        )
        from scripts.diagnostics.chronos2_temporal_producer_probe import BASE_OUTPUT_NAMES, OUTPUT_NAMES

        if file_sha(chronos2_encoder0_probe.__file__) != manifest["source_sha256"]["base_probe"]:
            raise ValueError("Probe source does not match reviewed manifest")
        if file_sha(chronos2_temporal_producer_probe.__file__) != manifest["source_sha256"]["probe"]:
            raise ValueError("Producer probe source does not match reviewed manifest")
        if file_sha(Path(__file__)) != manifest["source_sha256"]["runner"]:
            raise ValueError("Runner source does not match reviewed manifest")
        if manifest["original_final_device_sha256"] != ORIGINAL_FINAL_SHA256:
            raise ValueError("Original CI819 final identity anchor changed")
        library = Path(ffi_base._LIB._name).resolve()
        if file_sha(library) != manifest["runtime_library_sha256"]:
            raise ValueError("Runtime library differs from the original CI819 environment")
        result["runtime_library_sha256"] = file_sha(library)
        result["versions"] = {
            name: importlib.metadata.version(name)
            for name in ["rebel-compiler", "torch", "transformers", "chronos-forecasting"]
        }
        if result["versions"]["rebel-compiler"] != manifest["runtime_SDK"]:
            raise ValueError("Runtime SDK mismatch")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        inputs = torch.load(args.bundle / "exact-ci819-inputs.pt", weights_only=True)
        receipt = json.loads((args.bundle / "original-inputs.json").read_text())["inputs"]
        for name in ["hidden", "attention_mask", "group_time_mask"]:
            value = inputs[name]
            if list(value.shape) != receipt[name]["shape"] or str(value.dtype) != receipt[name]["dtype"]:
                raise ValueError("Original input interface changed: " + name)
            if tensor_sha256(value) != receipt[name]["sha256"]:
                raise ValueError("Original input bytes changed: " + name)
        model = rebel.RBLNCompiledModel(args.bundle / "encoder0_producers.rbln")
        event("create_reserved_runtime")
        runtime = model.create_runtime(tensor_type="pt", device=0)
        event("run_same_original_inputs")
        outputs = runtime(inputs["hidden"], inputs["attention_mask"], inputs["group_time_mask"])
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 10:
            raise ValueError("Expected five full original boundaries and five six-row producers")
        hashes = {}
        result["observed_output_shapes"] = {}
        retained = {}
        for name, value in zip(OUTPUT_NAMES, outputs, strict=True):
            expected_shape = [512, 577, 768] if name in BASE_OUTPUT_NAMES else [6, 577, 768]
            if list(value.shape) != expected_shape or value.dtype != torch.float32:
                raise ValueError("Augmented output interface mismatch: " + name)
            hashes[name] = tensor_sha256(value)
            result["observed_output_shapes"][name] = list(value.shape)
            retained[name] = {
                "active": value[:3].clone(),
                "selected_padded": (value[[3, 4, 511]] if name in BASE_OUTPUT_NAMES else value[3:]).clone(),
            }
        torch.save(retained, args.out / "selected-boundaries.pt")
        result["full_output_sha256"] = hashes
        (args.out / "first-output-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
        # This gate must precede any use of intermediate differences as original-graph evidence.
        require_original_final_identity(outputs[:5], manifest["original_final_device_sha256"])
        if {name: hashes[name] for name in BASE_OUTPUT_NAMES} != manifest["original_boundary_sha256"]:
            raise ValueError("An original CI861 boundary changed; no original-device attribution")
        del outputs
        event("original_final_bytes_exact_repeat_start")
        repeated = runtime(inputs["hidden"], inputs["attention_mask"], inputs["group_time_mask"])
        repeat_hashes = {name: tensor_sha256(value) for name, value in zip(OUTPUT_NAMES, repeated, strict=True)}
        result["repeat_output_sha256"] = repeat_hashes
        if repeat_hashes != hashes:
            raise ValueError("Instrumented repeat differs; no original-graph attribution")
        require_original_final_identity(repeated[:5], manifest["original_final_device_sha256"])
        del repeated
        for name in ["hidden", "attention_mask", "group_time_mask"]:
            if tensor_sha256(inputs[name]) != receipt[name]["sha256"]:
                raise ValueError("Runtime changed original input bytes: " + name)
        result["valid_for_original_encoder0_attribution"] = True
        references = torch.load(args.bundle / "cpu-boundary-references.pt", weights_only=True)
        producer_references = torch.load(args.bundle / "cpu-producer-references.pt", weights_only=True)
        if set(references) & set(producer_references):
            raise ValueError("CPU reference names overlap")
        references.update(producer_references)
        if set(references) != set(OUTPUT_NAMES):
            raise ValueError("Incomplete original CPU reference outputs")

        def metrics(actual, expected):
            actual, expected = actual.double().flatten(), expected.double().flatten()
            if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                raise ValueError("Nonfinite diagnostic values")
            rms = expected.square().mean().sqrt()
            error = (actual - expected).square().mean().sqrt()
            denom = (actual - actual.mean()).norm() * (expected - expected.mean()).norm()
            return {
                "nrmse": (error / rms).item() if rms else None,
                "rms_error": error.item(),
                "pcc": (torch.dot(actual - actual.mean(), expected - expected.mean()) / denom).item()
                if denom
                else None,
                "max_abs": (actual - expected).abs().max().item(),
            }

        result["boundaries"] = {}
        for name in OUTPUT_NAMES:
            result["boundaries"][name] = {
                part: metrics(retained[name][part], references[name][part]) for part in ["active", "selected_padded"]
            }
        result["metrics_scope"] = (
            "Same-input native FP32 CPU oracle; diagnostic only, not a faithful primitive emulator or full-model pass"
        )
        event("diagnostic_complete_original_identity_valid")
    except BaseException:
        result.update(phase="failed", error=traceback.format_exc())
        print(result["error"], flush=True)
    finally:
        result["seconds"] = time.monotonic() - started
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["phase"] != "diagnostic_complete_original_identity_valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
