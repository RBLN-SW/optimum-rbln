"""Compare finite-input SiLU accuracy candidates; neither model support nor adoption."""

import argparse
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

from scripts.diagnostics.flux2_text_gating import as_outputs
from scripts.diagnostics.flux2_transformer_frozen import metrics, require_scheduled_execution, sha256


MANIFEST_SHA = "9342476cf21818d8cc35be684bd0f4046bdb6492221219e88530b92b97f11acb"
SDK = "0.12.1.dev333+gf66b88a7"
VARIANTS = ["baseline", "minimum_exp"]
CASES = ["short_component", "full_original_prompt", "native_valid512", "negative_tail_sweep"]


def verify_bundle(directory):
    directory = Path(directory)
    if sha256(directory / "manifest.json") != MANIFEST_SHA:
        raise ValueError("SiLU accuracy manifest differs")
    manifest = json.loads((directory / "manifest.json").read_text())
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Frozen accuracy bundle has missing or unexpected files")
    for name, receipt in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Frozen files must be ordinary files inside the bundle")
        if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
            raise ValueError(f"Frozen accuracy checksum differs: {name}")
    expected = {
        "sdk": SDK,
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "target": "RBLN-CR13",
        "shape": [1, 512, 9728],
        "variants": VARIANTS,
        "probe_only": True,
        "not_adopted": True,
        "full_pipeline_npu_acceptance": False,
        "original_graph_causal_attribution": False,
        "implementation_candidates": ["minimum_exp"],
        "placement_audit_pending": False,
        "accuracy_target_nrmse": 0.005,
        "accuracy_target_pcc": 0.9999,
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Accuracy experiment contract differs")
    if [c["name"] for c in manifest["cases"]] != CASES:
        raise ValueError("Required actual native inputs and negative tail sweep differ")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("debug-flux2-silu-accuracy"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {
        "phase": "preflight",
        "full_pipeline_npu_acceptance": False,
        "original_graph_causal_attribution": False,
        "not_adopted": True,
        "placement_audit_pending": False,
        "diagnostic_completed": False,
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

            if importlib.metadata.version("rebel-compiler") != SDK:
                raise ValueError("Wrong SDK for SiLU accuracy prototypes")
            if sha256(Path(ffi_base._LIB._name).resolve()) != manifest["compiler_dso_sha256"]:
                raise ValueError("Loaded SDK library differs")
            if rebel.get_npu_name(0) != "RBLN-CR13":
                raise ValueError("Physical device differs")
            torch.set_num_threads(8)
            torch.set_num_interop_threads(1)
            result.update(phase="execute", rows=[], repeated={}, baseline831_identity={}, candidate_accuracy={})
            for variant in VARIANTS:
                path = args.artifacts / variant / "compiled_model.rbln"
                metadata = rebel.RBLNCompiledModel.inspect(path)
                if (
                    metadata["npu"] != "RBLN-CR13"
                    or metadata["num_devices"] != 1
                    or len(metadata["outputs"]) != 1
                    or metadata["inputs"] != [{"name": "hidden", "shape": [1, 512, 9728], "dtype": "float32"}]
                ):
                    raise ValueError("Accuracy prototype artifact ABI differs")
                runtime = rebel.Runtime(path, tensor_type="pt", device=0)
                values = {}
                for case in CASES:
                    reference = torch.load(args.artifacts / "references" / f"{case}.pt", weights_only=True)
                    inputs = -torch.abs(reference["input"]) if variant == "exp" else reference["input"]
                    expected = reference["exp"] if variant == "exp" else reference["silu"]
                    value = as_outputs(runtime(inputs), 1)[0].detach().clone()
                    torch.save(value, args.output / f"{variant}-{case}.pt")
                    values[case] = value
                    if variant == "baseline" and "previous831_activation" in reference:
                        result["baseline831_identity"][case] = torch.equal(value, reference["previous831_activation"])
                    row = {
                        "variant": variant,
                        "case": case,
                        "implementation_eligible": variant in manifest["implementation_candidates"],
                        **metrics(value, expected),
                    }
                    mask = reference["mask"].bool().squeeze(0)
                    row["valid"] = metrics(value[:, mask], expected[:, mask])
                    row["padding"] = metrics(value[:, ~mask], expected[:, ~mask]) if bool((~mask).any()) else None
                    row["probe_accuracy_target_passed"] = row["nrmse"] <= 0.005 and row["pcc"] >= 0.9999
                    result["rows"].append(row)
                    print(json.dumps(row), flush=True)
                    repeated = as_outputs(runtime(inputs), 1)[0]
                    result["repeated"][f"{variant}-{case}"] = torch.equal(value, repeated)
                    (args.output / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
                reference = torch.load(args.artifacts / "references/short_component.pt", weights_only=True)
                inputs = -torch.abs(reference["input"]) if variant == "exp" else reference["input"]
                returned = as_outputs(runtime(inputs), 1)[0]
                reloaded = rebel.Runtime(path, tensor_type="pt", device=0)
                fresh = as_outputs(reloaded(inputs), 1)[0]
                result["repeated"][f"{variant}-return-reload"] = torch.equal(
                    values["short_component"], returned
                ) and torch.equal(returned, fresh)
                del reloaded, runtime
                result["candidate_accuracy"][variant] = all(
                    row["probe_accuracy_target_passed"] for row in result["rows"] if row["variant"] == variant
                )
            if not all(result["repeated"].values()) or not all(result["baseline831_identity"].values()):
                raise ValueError("Baseline identity or accuracy runtime state changed values")
            if not result["candidate_accuracy"]["minimum_exp"]:
                raise ValueError("Minimum-exp candidate failed the original accuracy target")
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
