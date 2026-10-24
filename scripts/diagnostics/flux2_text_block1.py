"""Capture second-block features only when the complete native graph is unchanged."""

import argparse
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

from scripts.diagnostics.flux2_transformer_frozen import metrics, require_scheduled_execution, sha256


MANIFEST_SHA = "155054bd2129cd5e4139e7b36257d26b73937c237b070b4c999f9317aefbb564"
SDK = "0.12.1.dev333+gf66b88a7"
FEATURES = [
    "input_norm",
    "attention",
    "first_residual",
    "post_attention_norm",
    "gated_product",
    "mlp_output",
    "final_residual",
]


def verify_bundle(directory):
    directory = Path(directory)
    if sha256(directory / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Block diagnostic manifest differs")
    manifest = json.loads((directory / "manifest.json").read_text())
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Frozen block bundle has missing or unexpected files")
    for name, receipt in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Frozen files must be ordinary files inside the bundle")
        print(json.dumps({"phase": "verify_file", "file": name, "bytes": receipt["bytes"]}), flush=True)
        if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
            raise ValueError(f"Frozen block checksum differs: {name}")
    expected = {
        "sdk": SDK,
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "target": "RBLN-CR13",
        "batch": 1,
        "sequence_length": 512,
        "parameters": 4022468096,
        "decoder_layers": 36,
        "native_outputs": 37,
        "physical_outputs": 44,
        "block_index": 1,
        "feature_names": FEATURES,
        "observation_mechanism": "actual_native_producer_outputs",
        "required_frontend": "export",
        "diagnostic_only": True,
        "full_pipeline_npu_acceptance": False,
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Complete native model diagnostic contract differs")
    return manifest


def identity_checks(values, baseline):
    import torch

    if len(values) != 44 or set(baseline) != set(range(37)):
        raise ValueError("Native diagnostic output ABI differs")
    return {
        "native37": [torch.equal(values[i], baseline[i]) for i in range(37)],
        "auxiliary_final_matches_native_block": torch.equal(values[43], values[2]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("debug-flux2-text-block1-native-path"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {"phase": "preflight", "full_pipeline_npu_acceptance": False, "diagnostic_completed": False}
    code = 1
    try:
        if args.execute:
            require_scheduled_execution()
        manifest = verify_bundle(args.artifacts)
        if args.verify_only:
            result["phase"] = "verified_only"
            code = 0
        else:
            import rebel
            import torch
            from tvm._ffi import base as ffi_base

            if importlib.metadata.version("rebel-compiler") != SDK:
                raise ValueError("Wrong SDK for frozen block graph")
            if sha256(Path(ffi_base._LIB._name).resolve()) != manifest["compiler_dso_sha256"]:
                raise ValueError("Loaded SDK library differs")
            if rebel.get_npu_name(0) != "RBLN-CR13":
                raise ValueError("Physical device differs")
            torch.set_num_threads(8)
            torch.set_num_interop_threads(1)
            path = args.artifacts / "instrumented/compiled_model.rbln"
            metadata = rebel.RBLNCompiledModel.inspect(path)
            if metadata["npu"] != "RBLN-CR13" or metadata["num_devices"] != 1 or len(metadata["outputs"]) != 44:
                raise ValueError("Instrumented artifact differs")
            runtime = rebel.Runtime(path, tensor_type="pt", device=0)
            result.update(phase="execute", cases=[], feature_metrics=[])
            first_values = None
            for case in manifest["cases"]:
                name = case["name"]
                reference = torch.load(args.artifacts / "references" / f"{name}.pt", weights_only=True)
                baseline = torch.load(args.artifacts / "baseline" / f"{name}.pt", weights_only=True)
                features = torch.load(args.artifacts / "features" / f"{name}.pt", weights_only=True)["features"]
                inputs = reference["inputs"]
                values = tuple(v.detach().clone() for v in runtime(inputs["input_ids"], inputs["attention_mask"]))
                torch.save(values, args.output / f"{name}.pt")
                identities = identity_checks(values, baseline)
                repeated = runtime(inputs["input_ids"], inputs["attention_mask"])
                repeat_exact = all(torch.equal(a, b) for a, b in zip(values, repeated, strict=True))
                result["cases"].append({"case": name, "identities": identities, "repeat_exact": repeat_exact})
                # Capture all cases before rejecting changed instrumentation. Metrics alone never establish causality.
                mask = inputs["attention_mask"].bool().squeeze(0)
                for feature, value, expected in zip(FEATURES, values[37:], features, strict=True):
                    row = {"case": name, "feature": feature, **metrics(value, expected)}
                    row["valid"] = metrics(value[:, mask], expected[:, mask])
                    row["padding"] = metrics(value[:, ~mask], expected[:, ~mask]) if bool((~mask).any()) else None
                    result["feature_metrics"].append(row)
                    print(json.dumps(row), flush=True)
                if first_values is None:
                    first_values = values
                (args.output / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
            first = torch.load(args.artifacts / "references/short_component.pt", weights_only=True)["inputs"]
            returned = runtime(first["input_ids"], first["attention_mask"])
            reloaded = rebel.Runtime(path, tensor_type="pt", device=0)
            fresh = reloaded(first["input_ids"], first["attention_mask"])
            result["return_reload_exact"] = all(
                torch.equal(a, b) and torch.equal(a, c) for a, b, c in zip(first_values, returned, fresh, strict=True)
            )
            result["causal_identity_passed"] = all(
                all(c["identities"]["native37"]) and c["identities"]["auxiliary_final_matches_native_block"]
                for c in result["cases"]
            )
            if not result["causal_identity_passed"]:
                raise ValueError("Instrumentation changed native values; intermediate attribution rejected")
            if not result["return_reload_exact"] or not all(c["repeat_exact"] for c in result["cases"]):
                raise ValueError("Instrumented runtime state changed values")
            result.update(phase="diagnostic_finished", diagnostic_completed=True)
            code = 0
    except BaseException:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        result.update(exit_code=code, seconds=time.monotonic() - started)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
