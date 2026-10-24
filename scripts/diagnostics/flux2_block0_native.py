"""Check partial first-block probes against original full-model NPU endpoints."""

import argparse
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

from scripts.diagnostics.flux2_transformer_frozen import metrics, require_scheduled_execution, sha256


SDK = "0.12.1.dev336+g6a96bd1a"
MANIFEST_SHA = "a3829dc6c7887a385ed44b3df3ea5fe25136aaa841f0dbd6b213fadfd7a2b73e"
FEATURES = [
    "final_residual",
    "input_norm",
    "attention",
    "first_residual",
    "post_attention_norm",
    "gate_proj",
    "up_proj",
    "activation",
    "gated_product",
    "mlp_output",
]
FILES = {
    "native.rbln",
    "observed.rbln",
    "compile-result.json",
    "reference-result.json",
    "references/short_component.pt",
    "references/native_valid512.pt",
}


def same_bits(left, right):
    import torch

    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
    )


def identity_checks(native, observed, original):
    if len(observed) != len(FEATURES):
        raise ValueError("Partial block output ABI differs")
    return {
        "native_matches_full": same_bits(native, original),
        "observed_matches_full": same_bits(observed[0], original),
        "observed_matches_native": same_bits(observed[0], native),
    }


def verify_bundle(directory):
    directory = Path(directory)
    if sha256(directory / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Partial block manifest differs")
    manifest = json.loads((directory / "manifest.json").read_text())
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if set(manifest["files"]) != FILES or actual != FILES | {"manifest.json"}:
        raise ValueError("Partial bundle contains missing or unexpected files")
    for name, receipt in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Frozen files must be ordinary files inside the bundle")
        if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
            raise ValueError(f"Frozen block checksum differs: {name}")
    expected = {
        "sdk": SDK,
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "compiler_source_commit": "6a96bd1a7e1591fc98d96b8a5727ceb6bc67094b",
        "primitive_commit": "f9ad60764de5a48c331e739a1aa90cf500a43215",
        "target": "RBLN-CR13",
        "num_devices": 1,
        "batch": 1,
        "sequence_length": 512,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "block_index": 0,
        "probe_parameters": 100930816,
        "source_model_parameters": 4022468096,
        "source_decoder_layers": 36,
        "partial_block_probe": True,
        "diagnostic_only": True,
        "full_pipeline_npu_acceptance": False,
        "original_graph_causal_attribution": False,
        "feature_names": FEATURES,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Partial diagnostic contract differs")
    if [(c["name"], c["source_ci"]) for c in manifest["cases"]] != [
        ("short_component", 836),
        ("native_valid512", 837),
    ]:
        raise ValueError("Original full-model baseline cases differ")
    return manifest


def as_outputs(value, count):
    values = tuple(value) if isinstance(value, (tuple, list)) else (value,)
    if len(values) != count:
        raise ValueError("Partial runtime output ABI differs")
    return tuple(v.detach().clone() for v in values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("debug-flux2-block0"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "phase": "preflight",
        "full_pipeline_npu_acceptance": False,
        "original_graph_causal_attribution": False,
        "partial_block_probe": True,
    }
    started = time.monotonic()
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
                raise ValueError("Runtime SDK differs")
            if sha256(Path(ffi_base._LIB._name).resolve()) != manifest["compiler_dso_sha256"]:
                raise ValueError("Loaded SDK library differs")
            if rebel.get_npu_name(0) != "RBLN-CR13":
                raise ValueError("Physical target differs")
            runtimes = {}
            for name, count in [("native", 1), ("observed", len(FEATURES))]:
                path = args.artifacts / f"{name}.rbln"
                metadata = rebel.RBLNCompiledModel.inspect(path)
                if metadata["npu"] != "RBLN-CR13" or metadata["num_devices"] != 1 or len(metadata["outputs"]) != count:
                    raise ValueError("Partial binary ABI or target differs")
                runtimes[name] = rebel.Runtime(path, tensor_type="pt", device=0)
            result.update(phase="execute", cases=[], feature_metrics=[], intrinsic_metrics=[])
            for case in manifest["cases"]:
                name = case["name"]
                ref = torch.load(args.artifacts / "references" / f"{name}.pt", weights_only=True)
                inputs = ref["inputs"]
                native = as_outputs(runtimes["native"](inputs["hidden"], inputs["attention_mask"]), 1)[0]
                observed = as_outputs(runtimes["observed"](inputs["hidden"], inputs["attention_mask"]), len(FEATURES))
                torch.save({"native": native, "observed": observed}, args.output / f"{name}.pt")
                checks = identity_checks(native, observed, ref["original_full_NPU_h1"])
                repeated = as_outputs(runtimes["observed"](inputs["hidden"], inputs["attention_mask"]), len(FEATURES))
                fresh = rebel.Runtime(args.artifacts / "observed.rbln", tensor_type="pt", device=0)
                reloaded = as_outputs(fresh(inputs["hidden"], inputs["attention_mask"]), len(FEATURES))
                exact = all(
                    same_bits(a, b) and same_bits(a, c) for a, b, c in zip(observed, repeated, reloaded, strict=True)
                )
                result["cases"].append({"name": name, "identities": checks, "repeat_reload_exact": exact})
                mask = inputs["attention_mask"][0].bool()
                for feature, actual, expected in zip(FEATURES, observed, ref["CPU_features"], strict=True):
                    row = {
                        "case": name,
                        "feature": feature,
                        "diagnostic_graph_only": True,
                        **metrics(actual, expected),
                    }
                    row["valid"] = metrics(actual[:, mask], expected[:, mask])
                    row["padding"] = metrics(actual[:, ~mask], expected[:, ~mask]) if bool((~mask).any()) else None
                    result["feature_metrics"].append(row)
                    print(json.dumps(row), flush=True)
                for operation, actual, expected in [
                    ("silu", observed[7], torch.nn.functional.silu(observed[5])),
                    ("multiply", observed[8], observed[7] * observed[6]),
                ]:
                    row = {
                        "case": name,
                        "operation": operation,
                        "same_actual_producer_input": True,
                        "diagnostic_graph_only": True,
                        **metrics(actual, expected),
                    }
                    result["intrinsic_metrics"].append(row)
                    print(json.dumps(row), flush=True)
                (args.output / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
            if not all(all(c["identities"].values()) for c in result["cases"]):
                result["phase"] = "identity_failed"
                raise ValueError("Partial probe changed original endpoint; attribution rejected")
            if not all(c["repeat_reload_exact"] for c in result["cases"]):
                raise ValueError("Partial runtime state changed values")
            result.update(phase="diagnostic_finished", original_graph_causal_attribution=True)
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
