"""Localize native SDPA with packet-matched QK, masked softmax and PV components."""

import argparse
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

from scripts.diagnostics.flux2_block0_native import SDK, same_bits
from scripts.diagnostics.flux2_block0_selective import position_metrics
from scripts.diagnostics.flux2_sdpa_core import attention_chunks
from scripts.diagnostics.flux2_softmax_accuracy import create_runtime, from_blocked, to_blocked
from scripts.diagnostics.flux2_transformer_frozen import metrics, require_scheduled_execution, sha256


MANIFEST_SHA = "24b87bc4bfc614d47033222f091659c21f056c51bb508a3c37ae5c870eaed04d"
FILES = {
    "qk.rbln",
    "softmax.rbln",
    "pv.rbln",
    "compile-result.json",
    "packet-audit.json",
    "references/short_component.pt",
    "references/native_valid512.pt",
}


def verify_bundle(directory):
    directory = Path(directory)
    if sha256(directory / "manifest.json") != MANIFEST_SHA:
        raise ValueError("SDPA-components manifest differs")
    manifest = json.loads((directory / "manifest.json").read_text())
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if set(manifest["files"]) != FILES or actual != FILES | {"manifest.json"}:
        raise ValueError("SDPA-components file set differs")
    for name, receipt in manifest["files"].items():
        path = directory / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
            or path.stat().st_size != receipt["bytes"]
            or sha256(path) != receipt["sha256"]
        ):
            raise ValueError(f"Frozen SDPA-components checksum differs: {name}")
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
        "tiling_identity_verified": True,
        "physical_kernel_packets_match": True,
        "whole_graph_scheduling_identity_verified": False,
        "diagnostic_kind": "native_sdpa_components",
        "cases": ["short_component", "native_valid512"],
        "source_CIs": [836, 837, 841, 843, 844, 846],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("SDPA-components diagnostic contract differs")
    return manifest


def expanded_chunks(query, key, value):
    return [
        (q, k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1))
        for q, k, v in attention_chunks(query, key, value)
    ]


def output_tensor(value, shape, kind):
    import torch

    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{kind} output count differs")
        value = value[0]
    if value.shape != shape or value.dtype != torch.float32:
        raise ValueError(f"{kind} output ABI differs")
    return value


def run_components(runtimes, query, key, value, mask):
    logits = output_tensor(runtimes["qk"](query, key), (1, 8, 512, 512), "qk")
    blocked = output_tensor(
        runtimes["softmax"](to_blocked(logits), to_blocked(mask.float())), (1, 8, 8, 512, 64), "softmax"
    )
    probability = from_blocked(blocked)
    context = output_tensor(runtimes["pv"](probability, value), (1, 8, 512, 128), "pv")
    return logits, probability, context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("debug-flux2-sdpa-components"))
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
        "physical_kernel_packets_match": True,
        "whole_graph_scheduling_identity_verified": False,
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
            import math

            import rebel
            import torch
            from tvm._ffi import base as ffi_base

            if (
                importlib.metadata.version("rebel-compiler") != SDK
                or sha256(Path(ffi_base._LIB._name).resolve()) != manifest["compiler_dso_sha256"]
            ):
                raise ValueError("SDPA-components SDK differs")
            if rebel.get_npu_name(0) != "RBLN-CR13":
                raise ValueError("Physical target differs")
            kinds = ("qk", "softmax", "pv")
            for kind in kinds:
                metadata = rebel.RBLNCompiledModel.inspect(args.artifacts / f"{kind}.rbln")
                if metadata["npu"] != "RBLN-CR13" or metadata["num_devices"] != 1 or len(metadata["outputs"]) != 1:
                    raise ValueError(f"{kind} compiled ABI differs")
            runtimes = {kind: create_runtime(args.artifacts / f"{kind}.rbln") for kind in kinds}
            all_passed = True
            for case in manifest["cases"]:
                ref = torch.load(args.artifacts / "references" / f"{case}.pt", weights_only=True)
                mask = ref["mask"]
                if mask.dtype != torch.bool or mask.shape != (1, 1, 512, 512):
                    raise ValueError("Frozen native mask differs")
                captured = {name: [] for name in ("logits", "probability", "context")}
                stability = []
                references = {name: [] for name in ("qk", "softmax", "pv")}
                for q, k, v in expanded_chunks(ref["query"], ref["key"], ref["value"]):
                    fresh = {kind: create_runtime(args.artifacts / f"{kind}.rbln") for kind in kinds}
                    runs = [run_components(current, q, k, v, mask) for current in (runtimes, runtimes, fresh)]
                    stability.append(all(same_bits(a, b) and same_bits(a, c) for a, b, c in zip(*runs, strict=True)))
                    logits, probability, _context = runs[0]
                    for name, observed in zip(captured, runs[0], strict=True):
                        captured[name].append(observed)
                    references["qk"].append(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(128))
                    references["softmax"].append(
                        torch.softmax(logits.masked_fill(~mask, torch.finfo(torch.float32).min), dim=-1)
                    )
                    references["pv"].append(torch.matmul(probability, v))
                actual = {name: torch.cat(values, dim=1) for name, values in captured.items()}
                own = {
                    kind: metrics(actual[name], torch.cat(references[kind], dim=1))
                    for kind, name in zip(kinds, actual, strict=True)
                }
                comparison = metrics(actual["context"], ref["CPU_context"])
                comparison["valid"], comparison["padding"] = position_metrics(
                    actual["context"], ref["CPU_context"], ref["position_mask"], 2
                )
                row = {
                    "case": case,
                    "repeat_reload_exact": all(stability),
                    "matches_original_context": same_bits(actual["context"], ref["original_context"]),
                    "context_vs_CPU": comparison,
                    "context_vs_original": metrics(actual["context"], ref["original_context"]),
                    "own_same_input_metrics": own,
                }
                all_passed = (
                    all_passed
                    and row["repeat_reload_exact"]
                    and row["matches_original_context"]
                    and comparison["passed"]
                    and all(x["passed"] for x in own.values())
                )
                result["cases"].append(row)
                torch.save(actual, args.output / f"{case}.pt")
                print(json.dumps(row), flush=True)
            result["phase"] = (
                "faithful_decomposition_passed" if all_passed else "decomposition_identity_or_accuracy_failed"
            )
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
