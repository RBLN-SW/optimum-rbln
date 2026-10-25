"""Measure masked softmax error against CPU and ideal output-format rounding."""

import argparse
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

from scripts.diagnostics.flux2_block0_native import same_bits
from scripts.diagnostics.flux2_block0_selective import position_metrics
from scripts.diagnostics.flux2_transformer_frozen import metrics, require_scheduled_execution, sha256


SDK = "0.12.1.dev337+ge09f8a18"
MANIFEST_SHA = "2fa5b65daaccf61df73efdfe8119db02fa23448009bbdbbd0ed2861c0adb9665"
FILES = {
    "softmax.rbln",
    "corpus-result.json",
    "compile-result.json",
    "layout-audit.json",
    "references/short_component.pt",
    "references/native_valid512.pt",
}


def to_blocked(values):
    if values.ndim != 4 or values.shape[-1] % 64:
        raise ValueError("Expected complete 64-lane key blocks")
    n, h, q, k = values.shape
    return values.reshape(n, h, q, k // 64, 64).permute(0, 1, 3, 2, 4).contiguous()


def from_blocked(values):
    if values.ndim != 5 or values.shape[-1] != 64:
        raise ValueError("Expected a 64-lane blocked output")
    n, h, c, q, lane = values.shape
    return values.permute(0, 1, 3, 2, 4).contiguous().reshape(n, h, q, c * lane)


def quantization_metrics(actual, reference, ideal):
    import torch

    if (
        actual.shape != reference.shape
        or ideal.shape != reference.shape
        or any(value.dtype != torch.float32 for value in (actual, reference, ideal))
    ):
        raise ValueError("Softmax output ABI differs")
    return {
        "actual_vs_cpu": metrics(actual, reference),
        "rounded_ideal_vs_cpu": metrics(ideal, reference),
        "actual_vs_rounded_ideal": metrics(actual, ideal),
    }


def create_runtime(artifact):
    import rebel

    return rebel.Runtime(artifact, tensor_type="pt", device=0)


def verify_bundle(directory):
    directory = Path(directory)
    if sha256(directory / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Softmax-accuracy manifest differs")
    manifest = json.loads((directory / "manifest.json").read_text())
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if set(manifest["files"]) != FILES or actual != FILES | {"manifest.json"}:
        raise ValueError("Softmax-accuracy file set differs")
    for name, receipt in manifest["files"].items():
        path = directory / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
            or path.stat().st_size != receipt["bytes"]
            or sha256(path) != receipt["sha256"]
        ):
            raise ValueError(f"Frozen query-core checksum differs: {name}")
    expected = {
        "sdk": SDK,
        "compiler_source_commit": "e09f8a18b64491b1a3e1f47eaa6f8434c4cd4374",
        "primitive_commit": "6ba567512df2f86655601c3fe382823059a1057d",
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "target": "RBLN-CR13",
        "num_devices": 1,
        "batch": 1,
        "sequence_length": 512,
        "query_heads": 32,
        "heads_per_call": 8,
        "head_groups": 4,
        "key_length": 512,
        "lane_width": 64,
        "operator_only_probe": True,
        "full_pipeline_npu_acceptance": False,
        "original_graph_causal_attribution": False,
        "physical_op_signatures_match": True,
        "tiling_identity_verified": False,
        "derived_logits_not_original_capture": True,
        "input_is_exact_physical_value": True,
        "cases": ["short_component", "native_valid512"],
        "source_CIs": [841, 843, 844],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Softmax-accuracy diagnostic contract differs")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("debug-flux2-softmax-accuracy"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {
        "phase": "preflight",
        "operator_only_probe": True,
        "full_pipeline_npu_acceptance": False,
        "original_graph_causal_attribution": False,
        "tiling_identity_verified": False,
        "cases": [],
    }
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
                raise ValueError("Softmax-accuracy runtime SDK differs")
            if rebel.get_npu_name(0) != "RBLN-CR13":
                raise ValueError("Physical target differs")
            metadata = rebel.RBLNCompiledModel.inspect(args.artifacts / "softmax.rbln")
            if metadata["npu"] != "RBLN-CR13" or metadata["num_devices"] != 1 or len(metadata["outputs"]) != 1:
                raise ValueError("Softmax-accuracy compiled ABI differs")
            runtime = create_runtime(args.artifacts / "softmax.rbln")
            all_passed = True
            for case in manifest["cases"]:
                ref = torch.load(args.artifacts / "references" / f"{case}.pt", weights_only=True)
                if (
                    ref["logits"].shape != (1, 32, 512, 512)
                    or ref["mask"].shape != (1, 1, 512, 512)
                    or ref["mask"].dtype != torch.bool
                ):
                    raise ValueError("Frozen full native logits/mask contract differs")
                mask = to_blocked(ref["mask"].float())
                groups, stability = [], []
                for logits in ref["logits"].split(8, dim=1):
                    outputs = []
                    for current in [
                        runtime,
                        runtime,
                        create_runtime(args.artifacts / "softmax.rbln"),
                    ]:
                        observed = current(to_blocked(logits), mask)
                        if isinstance(observed, (list, tuple)):
                            if len(observed) != 1:
                                raise ValueError("Softmax output count differs")
                            observed = observed[0]
                        if observed.shape != (1, 8, 8, 512, 64) or observed.dtype != torch.float32:
                            raise ValueError("Softmax output ABI differs")
                        outputs.append(observed)
                    stability.append(same_bits(outputs[0], outputs[1]) and same_bits(outputs[0], outputs[2]))
                    groups.append(from_blocked(outputs[0]))
                if len(groups) != 4:
                    raise ValueError("All four query-head groups are required")
                actual = torch.cat(groups, dim=1)
                comparison = quantization_metrics(actual, ref["CPU_probability"], ref["ideal_physical_probability"])
                comparison["actual_vs_cpu"]["valid"], comparison["actual_vs_cpu"]["padding"] = position_metrics(
                    actual, ref["CPU_probability"], ref["position_mask"], 2
                )
                row = {
                    "case": case,
                    "repeat_reload_exact": all(stability),
                    "metrics": comparison,
                    "derived_logits_not_original_capture": True,
                }
                all_passed = all_passed and row["repeat_reload_exact"] and comparison["actual_vs_cpu"]["passed"]
                result["cases"].append(row)
                torch.save({"probability": actual}, args.output / f"{case}.pt")
                print(json.dumps(row), flush=True)
            result["phase"] = "operator_diagnostic_passed" if all_passed else "operator_diagnostic_failed"
            code = 0 if all_passed else 1
    except BaseException:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        result.update(exit_code=code, seconds=time.monotonic() - started)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
