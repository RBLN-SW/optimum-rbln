"""Diagnose native-shaped masked SDPA on captured query/key/value operands."""

import argparse
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

from scripts.diagnostics.flux2_block0_native import SDK, same_bits
from scripts.diagnostics.flux2_block0_selective import position_metrics
from scripts.diagnostics.flux2_query_core import checked_output, join_groups
from scripts.diagnostics.flux2_transformer_frozen import metrics, require_scheduled_execution, sha256


MANIFEST_SHA = "f6b4589981d3d961fdb6864e897e641e00a4005a00673b1742957679165330f9"
FILES = {
    "sdpa.rbln",
    "compile-result.json",
    "layout-audit.json",
    "references/short_component.pt",
    "references/native_valid512.pt",
}


def attention_chunks(query, key, value):
    import torch

    if (
        query.shape != (1, 32, 512, 128)
        or key.shape != (1, 8, 512, 128)
        or value.shape != key.shape
        or any(tensor.dtype != torch.float32 for tensor in (query, key, value))
    ):
        raise ValueError("Full native Q/K/V operands are required")
    return [
        (
            query[:, group * 8 : (group + 1) * 8].contiguous(),
            key[:, group * 2 : (group + 1) * 2].contiguous(),
            value[:, group * 2 : (group + 1) * 2].contiguous(),
        )
        for group in range(4)
    ]


def verify_bundle(directory):
    directory = Path(directory)
    if sha256(directory / "manifest.json") != MANIFEST_SHA:
        raise ValueError("SDPA-core manifest differs")
    manifest = json.loads((directory / "manifest.json").read_text())
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if set(manifest["files"]) != FILES or actual != FILES | {"manifest.json"}:
        raise ValueError("SDPA-core file set differs")
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
        "q_heads_per_call": 8,
        "kv_heads_per_call": 2,
        "kv_heads": 8,
        "head_groups": 4,
        "operator_only_probe": True,
        "full_pipeline_npu_acceptance": False,
        "original_graph_causal_attribution": False,
        "physical_op_signatures_match": True,
        "tiling_identity_verified": False,
        "cases": ["short_component", "native_valid512"],
        "source_CIs": [836, 837, 841, 843],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("SDPA-core diagnostic contract differs")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("debug-flux2-sdpa-core"))
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
                raise ValueError("SDPA-core runtime SDK differs")
            if rebel.get_npu_name(0) != "RBLN-CR13":
                raise ValueError("Physical target differs")
            metadata = rebel.RBLNCompiledModel.inspect(args.artifacts / "sdpa.rbln")
            if metadata["npu"] != "RBLN-CR13" or metadata["num_devices"] != 1 or len(metadata["outputs"]) != 1:
                raise ValueError("SDPA-core compiled ABI differs")
            runtime = rebel.Runtime(args.artifacts / "sdpa.rbln", tensor_type="pt", device=0)
            all_passed = True
            for case in manifest["cases"]:
                ref = torch.load(args.artifacts / "references" / f"{case}.pt", weights_only=True)
                if ref["mask"].dtype != torch.bool or ref["mask"].shape != (1, 1, 512, 512):
                    raise ValueError("Frozen export boolean mask differs")
                groups, stability = [], []
                for query, key, value in attention_chunks(ref["query"], ref["key"], ref["value"]):
                    outputs = []
                    for current in [
                        runtime,
                        runtime,
                        rebel.Runtime(args.artifacts / "sdpa.rbln", tensor_type="pt", device=0),
                    ]:
                        observed = current(query, key, value, ref["mask"])
                        if isinstance(observed, (list, tuple)):
                            if len(observed) != 1:
                                raise ValueError("SDPA-core output count differs")
                            observed = observed[0]
                        outputs.append(checked_output(observed, "sdpa"))
                    stability.append(same_bits(outputs[0], outputs[1]) and same_bits(outputs[0], outputs[2]))
                    groups.append(outputs[0])
                actual = join_groups(groups, "sdpa")
                comparison = metrics(actual, ref["CPU_context"])
                comparison["valid"], comparison["padding"] = position_metrics(
                    actual, ref["CPU_context"], ref["position_mask"], 2
                )
                row = {
                    "case": case,
                    "repeat_reload_exact": all(stability),
                    "matches_original_context": same_bits(actual, ref["original_context"]),
                    "metrics": comparison,
                    "original_context_metrics": metrics(actual, ref["original_context"]),
                }
                all_passed = all_passed and row["repeat_reload_exact"] and comparison["passed"]
                result["cases"].append(row)
                torch.save({"context": actual}, args.output / f"{case}.pt")
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
