"""Run captured-input operator diagnostics only in a reserved scheduler CA25 job."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
import traceback
from pathlib import Path


ROWS = [0, 1, 2, 3, 4, 511]
CASES = [("sdpa_b1", 1, False), ("sdpa_b6", 6, False), ("explicit_b6", 6, True)]
FILES = {
    "sdpa_b1.rbln",
    "sdpa_b6.rbln",
    "explicit_b6.rbln",
    "inputs.pt",
    "goldens.pt",
    "CI862-observed-context.pt",
    "preparation65.json",
    "compile-result65.json",
    "placement-audit65.json",
    "initgen-binding65.json",
    "initgen-aggregate65.json",
    "sdpa_b1-initgen.json",
    "sdpa_b6-initgen.json",
    "explicit_b6-initgen.json",
    "original-ci862-result.json",
    "attention_probe.py",
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def verify_bundle(root, expected_manifest_sha):
    root = Path(root).resolve()
    manifest_file = root / "manifest.json"
    if manifest_file.is_symlink() or file_sha(manifest_file) != expected_manifest_sha:
        raise ValueError("Manifest SHA mismatch or symlink")
    manifest = json.loads(manifest_file.read_text())
    if set(manifest["files"]) != FILES:
        raise ValueError("Unexpected frozen bundle membership")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != FILES | {"manifest.json"}:
        raise ValueError("Missing or extra frozen bundle files")
    for name, row in manifest["files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Frozen file is a symlink or missing: " + name)
        if path.stat().st_size != row["bytes"] or file_sha(path) != row["sha256"]:
            raise ValueError("Frozen file size/SHA mismatch: " + name)
    if (
        manifest["model"] != "amazon/chronos-2"
        or manifest["revision"] != "29ec3766d36d6f73f0696f85560a422f50e8498c"
        or manifest["target"] != "RBLN-CA25"
        or manifest["full_model_acceptance"] is not False
        or manifest["original_device_attribution"] is not False
        or manifest["original_identity_gate_failed"] is not True
    ):
        raise ValueError("Invalid model or diagnostic-only scope")
    if manifest["observation_rows"] != ROWS:
        raise ValueError("Captured row mapping changed")
    if [(row["name"], row["batch"], row["explicit"]) for row in manifest["cases"]] != CASES:
        raise ValueError("Operator case geometry changed")
    if manifest["criteria"] != {"nrmse_max": 0.05, "pcc_min": 0.99, "repeat_count": 10}:
        raise ValueError("Diagnostic criteria changed")
    return manifest


def verify_tensors(root, manifest):
    import torch

    inputs = torch.load(root / "inputs.pt", weights_only=True, map_location="cpu")
    goldens = torch.load(root / "goldens.pt", weights_only=True, map_location="cpu")
    captured = torch.load(root / "CI862-observed-context.pt", weights_only=True, map_location="cpu")
    ci862 = json.loads((root / "original-ci862-result.json").read_text())
    if ci862["valid_for_original_encoder0_attribution"] is not False or ci862["phase"] != "failed":
        raise ValueError("Expected the explicitly invalid-identity source variant")
    expected_shapes = {name: [6, 12, 577, 64] for name in ["query", "key", "value"]}
    expected_shapes["attention_mask"] = [6, 1, 1, 577]
    if set(inputs) != set(expected_shapes) or set(goldens) != {name for name, _, _ in CASES}:
        raise ValueError("Input or golden names changed")
    before_hashes = {}
    for name, value in inputs.items():
        if list(value.shape) != expected_shapes[name] or value.dtype != torch.float32:
            raise ValueError("Input shape/dtype changed: " + name)
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite input: " + name)
        before_hashes[name] = tensor_sha(value)
    if before_hashes != manifest["input_raw_sha256"]:
        raise ValueError("Input tensor bytes differ from frozen preparation")
    for name, source in [("query", "time0_q_rope"), ("key", "time0_k_rope"), ("value", "time0_v")]:
        unheaded = inputs[name].transpose(1, 2).reshape(6, 577, 768)
        if tensor_sha(unheaded) != ci862["full_output_sha256"][source]:
            raise ValueError("Input producer bytes not bound to actual CI862: " + name)
    if (
        list(captured.shape) != [6, 577, 768]
        or captured.dtype != torch.float32
        or tensor_sha(captured) != ci862["full_output_sha256"]["time0_attention_context"]
    ):
        raise ValueError("Captured context is not bound to actual CI862")
    for case in manifest["cases"]:
        golden = goldens[case["name"]]
        if (
            list(golden.shape) != [case["batch"], 577, 768]
            or golden.dtype != torch.float32
            or not torch.isfinite(golden).all()
            or tensor_sha(golden) != case["golden_raw_sha256"]
        ):
            raise ValueError("CPU golden changed: " + case["name"])
    return inputs, goldens, captured, before_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-only", action="store_true")
    modes.add_argument("--verify-tensors-only", action="store_true")
    args = parser.parse_args()
    manifest = verify_bundle(args.bundle, args.manifest_sha256)
    if args.verify_only:
        print(json.dumps({"verified_files": len(FILES), "tensor_binding_checked": False, "NPU_API_calls": 0}))
        return
    if args.verify_tensors_only:
        _, _, _, hashes = verify_tensors(args.bundle, manifest)
        print(json.dumps({"captured_input_binding_checked": True, "input_sha256": hashes, "NPU_API_calls": 0}))
        return
    if not os.environ.get("BUILDKITE_JOB_ID") or os.environ.get("RBLN_TEST_NPU") != "RBLN-CA25":
        raise ValueError("Execution requires the reserved scheduler CA25 job")
    if os.environ.get("RBLN_DUMMY_DEVICE") or os.environ.get("TVM_LIBRARY_PATH"):
        raise ValueError("Frozen execution forbids dummy devices and library overrides")
    if file_sha(Path(__file__)) != manifest["runner_sha256"]:
        raise ValueError("Runner source does not match the frozen manifest")
    if args.out is None:
        raise ValueError("Execution requires an output directory")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {
        "phase": "preflight",
        "full_model_acceptance": False,
        "original_device_attribution": False,
        "original_identity_gate_failed": True,
        "scope": "Reduced operator controls on CI862 variant inputs, not the original full512 graph",
        "manifest_sha256": args.manifest_sha256,
        "cases": [],
    }

    def event(phase):
        result.update(phase=phase, seconds=time.monotonic() - started)
        (args.out / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
        print(phase, flush=True)

    try:
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

        def metric(value, reference):
            value, reference = value.double().flatten(), reference.double().flatten()
            error = (value - reference).square().sum()
            energy = reference.square().sum()
            nrmse = (error / energy).sqrt().item() if energy else (0.0 if error == 0 else None)
            centered_value, centered_reference = value - value.mean(), reference - reference.mean()
            denominator = centered_value.norm() * centered_reference.norm()
            pcc = (centered_value @ centered_reference / denominator).item() if denominator else None
            return {"nrmse": nrmse, "pcc": pcc, "max_abs": (value - reference).abs().max().item()}

        inputs, goldens, captured, before_hashes = verify_tensors(args.bundle, manifest)
        result["actual_CI862_producer_bytes_bound"] = True
        result["input_sha256_before"] = before_hashes
        import rebel
        from tvm._ffi import base as ffi_base

        version = importlib.metadata.version("rebel-compiler")
        library_sha = file_sha(Path(ffi_base._LIB._name).resolve())
        if version != manifest["runtime_SDK"] or library_sha != manifest["runtime_library_sha256"]:
            raise ValueError("Frozen runtime SDK/library mismatch")
        result.update(runtime_SDK=version, runtime_library_sha256=library_sha)
        retained = {}
        for name, batch, _ in CASES:
            model = rebel.RBLNCompiledModel(args.bundle / (name + ".rbln"))
            metadata = rebel.RBLNCompiledModel.inspect(args.bundle / (name + ".rbln"))
            if metadata["npu"] != "RBLN-CA25" or metadata["num_devices"] != 1:
                raise ValueError("Unexpected compiled target")
            if len(metadata["outputs"]) != 1 or metadata["outputs"][0]["shape"] != [batch, 577, 768]:
                raise ValueError("Compiled output geometry changed")
            selected = [inputs[key][:batch].contiguous() for key in ["query", "key", "value", "attention_mask"]]

            def output(runtime, selected=selected, batch=batch):
                value = runtime(*selected)
                if isinstance(value, (tuple, list)):
                    if len(value) != 1:
                        raise ValueError("Expected exactly one operator output")
                    value = value[0]
                if list(value.shape) != [batch, 577, 768] or value.dtype != torch.float32:
                    raise ValueError("Runtime output shape/dtype changed")
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite operator output")
                return value.detach().cpu().clone()

            event("run_" + name)
            runtime = model.create_runtime(tensor_type="pt", device=0)
            first = output(runtime)
            first_hash = tensor_sha(first)
            repeat_hashes = [tensor_sha(output(runtime)) for _ in range(10)]
            repeats_exact = all(value == first_hash for value in repeat_hashes)
            reloaded = rebel.RBLNCompiledModel(args.bundle / (name + ".rbln"))
            reload_exact = torch.equal(first, output(reloaded.create_runtime(tensor_type="pt", device=0)))
            parity = metric(first, goldens[name])
            passed = (
                parity["nrmse"] is not None
                and parity["nrmse"] <= 0.05
                and parity["pcc"] is not None
                and parity["pcc"] >= 0.99
                and repeats_exact
                and reload_exact
            )
            result["cases"].append(
                {
                    "name": name,
                    "batch": batch,
                    "vs_own_CPU_golden": parity,
                    "vs_CI862_variant_context_diagnostic_only": metric(first, captured[:batch]),
                    "output_sha256": tensor_sha(first),
                    "repeat_output_sha256": repeat_hashes,
                    "ten_repeats_exact": repeats_exact,
                    "artifact_reload_exact": reload_exact,
                    "operator_criteria_passed": passed,
                }
            )
            retained[name] = first
            torch.save(retained, args.out / "outputs.pt")
        after_hashes = {name: tensor_sha(value) for name, value in inputs.items()}
        result["input_sha256_after"] = after_hashes
        result["input_postrun_hashes_exact"] = after_hashes == before_hashes
        result["operator_criteria_passed"] = all(row["operator_criteria_passed"] for row in result["cases"])
        result["operator_criteria_passed"] &= result["input_postrun_hashes_exact"]
        event("passed" if result["operator_criteria_passed"] else "operator_parity_failed")
    except Exception:
        result.update(phase="failed", error=traceback.format_exc())
        print(result["error"], flush=True)
    finally:
        result["seconds"] = time.monotonic() - started
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["phase"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
