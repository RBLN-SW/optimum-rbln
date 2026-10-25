"""Map selective observations while preserving strict original endpoint identity."""

import argparse
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

from scripts.diagnostics.flux2_block0_native import FEATURES as BLOCK_FEATURES
from scripts.diagnostics.flux2_block0_native import SDK, as_outputs, same_bits
from scripts.diagnostics.flux2_transformer_frozen import metrics, require_scheduled_execution, sha256


MANIFEST_SHA = "6b65453172de6834907a9ee7b057dd85bf1622be6e71a659d5f5369efb636599"
FEATURES = [
    *BLOCK_FEATURES,
    "query_rotary",
    "key_rotary",
    "value_heads",
    "attention_context",
    "query_projected",
    "query_normalized",
    "query_direct",
]
VARIANTS = {"query_projected": 14, "query_normalized": 15, "query_direct": 16}
SEQUENCE_AXES = {"query_projected": 1, "query_normalized": 1, "query_direct": 2}
FILES = {
    "native.rbln",
    "baseline-compile-result.json",
    "reference-result.json",
    "compile-result.json",
    "references/short_component.pt",
    "references/native_valid512.pt",
} | {name + ".rbln" for name in VARIANTS}


def faithful_cases(cases):
    if len(cases) != 2 or {case["name"] for case in cases} != {"short_component", "native_valid512"}:
        raise ValueError("Both original baseline cases are required")
    return all(case["matches_original"] is True and case["repeat_exact"] is True for case in cases)


def validate_outputs(variant, values, reference):
    index = VARIANTS[variant]
    expected = [reference["original_full_NPU_h1"]]
    if index is not None:
        expected.append(reference["CPU_features"][index])
    if len(values) != len(expected) or any(
        actual.shape != target.shape or actual.dtype != target.dtype
        for actual, target in zip(values, expected, strict=True)
    ):
        raise ValueError("Selective producer output ABI differs from frozen reference")


def position_metrics(actual, expected, mask, sequence_axis):
    if (
        sequence_axis not in {1, 2}
        or actual.ndim <= sequence_axis
        or actual.shape != expected.shape
        or mask.ndim != 1
        or actual.shape[sequence_axis] != mask.numel()
    ):
        raise ValueError("Declared sequence axis differs from the original position mask")
    actual = actual.movedim(sequence_axis, 1)
    expected = expected.movedim(sequence_axis, 1)
    valid = metrics(actual[:, mask], expected[:, mask])
    padding = metrics(actual[:, ~mask], expected[:, ~mask]) if bool((~mask).any()) else None
    return valid, padding


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
    if manifest.get("experiment") != "selective_observation_endpoint_identity" or manifest.get("variants") != VARIANTS:
        raise ValueError("Selective observation contract differs")
    if manifest.get("feature_sequence_axes") != SEQUENCE_AXES:
        raise ValueError("Producer sequence axes differ from the frozen contract")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("debug-flux2-selective"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-only", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {"phase": "preflight", "full_pipeline_npu_acceptance": False, "original_graph_causal_attribution": False}
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

            if (
                importlib.metadata.version("rebel-compiler") != SDK
                or sha256(Path(ffi_base._LIB._name).resolve()) != manifest["compiler_dso_sha256"]
            ):
                raise ValueError("Runtime SDK differs")
            if rebel.get_npu_name(0) != "RBLN-CR13":
                raise ValueError("Physical target differs")
            runtimes = {}
            for name, count in [("native", 1)] + [
                (name, 1 if index is None else 2) for name, index in VARIANTS.items()
            ]:
                path = args.artifacts / f"{name}.rbln"
                meta = rebel.RBLNCompiledModel.inspect(path)
                if meta["npu"] != "RBLN-CR13" or meta["num_devices"] != 1 or len(meta["outputs"]) != count:
                    raise ValueError("Selective runtime ABI differs")
                runtimes[name] = rebel.Runtime(path, tensor_type="pt", device=0)
            records = {name: [] for name in VARIANTS}
            result.update(phase="execute", cases=records, feature_metrics=[])
            for case in manifest["cases"]:
                name = case["name"]
                ref = torch.load(args.artifacts / "references" / f"{name}.pt", weights_only=True)
                inputs = ref["inputs"]
                original = ref["original_full_NPU_h1"]
                native = as_outputs(runtimes["native"](inputs["hidden"], inputs["attention_mask"]), 1)[0]
                actuals = {"native": native}
                for variant, index in VARIANTS.items():
                    count = 1 if index is None else 2
                    values = as_outputs(runtimes[variant](inputs["hidden"], inputs["attention_mask"]), count)
                    validate_outputs(variant, values, ref)
                    actuals[variant] = values
                    repeated = as_outputs(runtimes[variant](inputs["hidden"], inputs["attention_mask"]), count)
                    fresh = rebel.Runtime(args.artifacts / f"{variant}.rbln", tensor_type="pt", device=0)
                    reloaded = as_outputs(fresh(inputs["hidden"], inputs["attention_mask"]), count)
                    exact = all(
                        same_bits(a, b) and same_bits(a, c) for a, b, c in zip(values, repeated, reloaded, strict=True)
                    )
                    matched = same_bits(native, original) and same_bits(values[0], original)
                    record = {
                        "name": name,
                        "matches_original": matched,
                        "repeat_exact": exact,
                        "native_matches_full": same_bits(native, original),
                        "endpoint_metrics": metrics(values[0], original),
                    }
                    records[variant].append(record)
                    print(json.dumps({"variant": variant, **record}), flush=True)
                    if index is not None:
                        row = {
                            "case": name,
                            "variant": variant,
                            "diagnostic_graph_only": True,
                            **metrics(values[1], ref["CPU_features"][index]),
                        }
                        mask = inputs["attention_mask"][0].bool()
                        row["valid"], row["padding"] = position_metrics(
                            values[1], ref["CPU_features"][index], mask, SEQUENCE_AXES[variant]
                        )
                        result["feature_metrics"].append(row)
                torch.save(actuals, args.output / f"{name}.pt")
                (args.output / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
            result["attribution_by_variant"] = {name: faithful_cases(cases) for name, cases in records.items()}
            for row in result["feature_metrics"]:
                row["original_graph_causal_attribution"] = result["attribution_by_variant"][row["variant"]]
            if not all(result["attribution_by_variant"].values()):
                result["phase"] = "identity_failed"
                raise ValueError("Some observations changed original outputs; only matching variants are attributable")
            result["phase"] = "diagnostic_finished"
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
