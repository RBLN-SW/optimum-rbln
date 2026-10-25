"""Diagnose native-shaped Qwen3 RMSNorm/RoPE on captured query projections."""

import argparse
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

from scripts.diagnostics.flux2_block0_native import SDK, same_bits
from scripts.diagnostics.flux2_block0_selective import position_metrics
from scripts.diagnostics.flux2_transformer_frozen import metrics, require_scheduled_execution, sha256


MANIFEST_SHA = "79d2c7daa2c970f25ac90dcdac8d4492051407ed2d51d2d486f500b298f4fe43"
FILES = {
    "norm.rbln",
    "rope.rbln",
    "joint.rbln",
    "compile-result.json",
    "layout-audit.json",
    "references/short_component.pt",
    "references/native_valid512.pt",
}


def checked_output(value, kind):
    import torch

    shape = (1, 512, 8, 128) if kind == "norm" else (1, 8, 512, 128)
    if value.shape != shape or value.dtype != torch.float32:
        raise ValueError("Query-core output ABI differs")
    return value


def projected_chunks(projected):
    if projected.shape != (1, 512, 4096):
        raise ValueError("Full native projected query is required")
    return [part.contiguous().flatten(2) for part in projected.view(1, 512, 32, 128).split(8, dim=2)]


def join_groups(parts, kind):
    import torch

    if len(parts) != 4:
        raise ValueError("All four head groups are required")
    return torch.cat([checked_output(part, kind) for part in parts], dim=2 if kind == "norm" else 1)


def rotary_reference(normalized, cos, sin):
    import torch

    query = normalized.transpose(1, 2)
    half = query.shape[-1] // 2
    rotated = torch.cat((-query[..., half:], query[..., :half]), dim=-1)
    return query * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


def verify_bundle(directory):
    directory = Path(directory)
    if sha256(directory / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Query-core manifest differs")
    manifest = json.loads((directory / "manifest.json").read_text())
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if set(manifest["files"]) != FILES or actual != FILES | {"manifest.json"}:
        raise ValueError("Query-core file set differs")
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
        "compiler_source_commit": "6a96bd1a7e1591fc98d96b8a5727ceb6bc67094b",
        "primitive_commit": "f9ad60764de5a48c331e739a1aa90cf500a43215",
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "target": "RBLN-CR13",
        "num_devices": 1,
        "batch": 1,
        "sequence_length": 512,
        "query_heads": 32,
        "head_dim": 128,
        "heads_per_call": 8,
        "head_groups": 4,
        "operator_only_probe": True,
        "full_pipeline_npu_acceptance": False,
        "original_graph_causal_attribution": False,
        "physical_op_signatures_match": True,
        "tiling_identity_verified": False,
        "cases": ["short_component", "native_valid512"],
        "source_CIs": [836, 837, 841, 842],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Query-core diagnostic contract differs")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("debug-flux2-query-core"))
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
                raise ValueError("Query-core runtime SDK differs")
            if rebel.get_npu_name(0) != "RBLN-CR13":
                raise ValueError("Physical target differs")
            runtimes = {
                name: rebel.Runtime(args.artifacts / f"{name}.rbln", tensor_type="pt", device=0)
                for name in ["norm", "rope", "joint"]
            }
            all_passed = True
            for case in manifest["cases"]:
                ref = torch.load(args.artifacts / "references" / f"{case}.pt", weights_only=True)
                repeated_exact = []

                def run(name, value, stability):
                    outputs = []
                    for runtime in [
                        runtimes[name],
                        runtimes[name],
                        rebel.Runtime(args.artifacts / f"{name}.rbln", tensor_type="pt", device=0),
                    ]:
                        actual = runtime(value)
                        if isinstance(actual, (tuple, list)):
                            if len(actual) != 1:
                                raise ValueError("Query-core output count differs")
                            actual = actual[0]
                        outputs.append(checked_output(actual, name))
                    stability.append(same_bits(outputs[0], outputs[1]) and same_bits(outputs[0], outputs[2]))
                    return outputs[0]

                projected = projected_chunks(ref["projected"])
                norms = [run("norm", value, repeated_exact) for value in projected]
                normalized = join_groups(norms, "norm")
                composed = join_groups([run("rope", value, repeated_exact) for value in norms], "rope")
                teacher_forced = join_groups(
                    [run("rope", part.contiguous(), repeated_exact) for part in ref["CPU_normalized"].split(8, dim=2)],
                    "rope",
                )
                joint = join_groups([run("joint", value, repeated_exact) for value in projected], "joint")
                actual = {
                    "normalized": normalized,
                    "composed": composed,
                    "teacher_forced_rope": teacher_forced,
                    "joint": joint,
                }
                row = {
                    "case": case,
                    "repeat_reload_exact": all(repeated_exact),
                    "composed_joint_exact": same_bits(composed, joint),
                    "metrics": {
                        "norm": metrics(normalized, ref["CPU_normalized"]),
                        "rope_same_actual_norm": metrics(
                            composed, rotary_reference(normalized, ref["cos"], ref["sin"])
                        ),
                        "rope_same_CPU_norm": metrics(teacher_forced, ref["CPU_rotated"]),
                        "joint": metrics(joint, ref["CPU_rotated"]),
                        "composed": metrics(composed, ref["CPU_rotated"]),
                    },
                }
                sources = {
                    "norm": (normalized, ref["CPU_normalized"], 1),
                    "rope_same_actual_norm": (composed, rotary_reference(normalized, ref["cos"], ref["sin"]), 2),
                    "rope_same_CPU_norm": (teacher_forced, ref["CPU_rotated"], 2),
                    "joint": (joint, ref["CPU_rotated"], 2),
                    "composed": (composed, ref["CPU_rotated"], 2),
                }
                for name, (observed, expected, axis) in sources.items():
                    row["metrics"][name]["valid"], row["metrics"][name]["padding"] = position_metrics(
                        observed, expected, ref["mask"], axis
                    )
                all_passed = (
                    all_passed
                    and row["repeat_reload_exact"]
                    and all(value["passed"] for value in row["metrics"].values())
                )
                result["cases"].append(row)
                torch.save(actual, args.output / f"{case}.pt")
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
