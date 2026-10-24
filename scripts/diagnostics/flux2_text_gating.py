"""Probe trained second-block gating; changed-graph evidence is not model acceptance."""

import argparse
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

from scripts.diagnostics.flux2_transformer_frozen import metrics, require_scheduled_execution, sha256


MANIFEST_SHA = "50f3ee8922ff34627222254cdc553afef4c9d15ddc92232fb1cbeb6af793dc96"
SDK = "0.12.1.dev333+gf66b88a7"


def verify_bundle(directory):
    directory = Path(directory)
    if sha256(directory / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Gating probe manifest differs")
    manifest = json.loads((directory / "manifest.json").read_text())
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Frozen gating bundle has missing or unexpected files")
    for name, receipt in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Frozen files must be ordinary files inside the bundle")
        print(json.dumps({"phase": "verify_file", "file": name, "bytes": receipt["bytes"]}), flush=True)
        if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
            raise ValueError(f"Frozen gating checksum differs: {name}")
    expected = {
        "sdk": SDK,
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "target": "RBLN-CR13",
        "batch": 1,
        "sequence_length": 512,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "block_index": 1,
        "probe_only": True,
        "full_pipeline_npu_acceptance": False,
        "original_graph_causal_attribution": False,
        "source_graph": "CI830 instrumented graph; original identity check FAILED",
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Gating probe contract differs")
    return manifest


def as_outputs(value, count):
    values = tuple(value) if isinstance(value, (tuple, list)) else (value,)
    if len(values) != count:
        raise ValueError("Gating probe output count differs")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("debug-flux2-text-gating"))
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
                raise ValueError("Wrong SDK for trained gating probes")
            if sha256(Path(ffi_base._LIB._name).resolve()) != manifest["compiler_dso_sha256"]:
                raise ValueError("Loaded SDK library differs")
            if rebel.get_npu_name(0) != "RBLN-CR13":
                raise ValueError("Physical device differs")
            torch.set_num_threads(8)
            torch.set_num_interop_threads(1)
            runtimes = {}
            for name, count in [("product", 1), ("observed", 4), ("activation", 1)]:
                path = args.artifacts / name / "compiled_model.rbln"
                metadata = rebel.RBLNCompiledModel.inspect(path)
                width = 9728 if name == "activation" else 2560
                if (
                    metadata["npu"] != "RBLN-CR13"
                    or metadata["num_devices"] != 1
                    or len(metadata["outputs"]) != count
                    or metadata["inputs"] != [{"name": "hidden", "shape": [1, 512, width], "dtype": "float32"}]
                ):
                    raise ValueError("Trained probe artifact ABI differs")
                runtimes[name] = rebel.Runtime(path, tensor_type="pt", device=0)
            result.update(phase="execute", cases=[], metrics=[], repeated={})
            first_inputs = {}
            first_outputs = {}
            for case in manifest["cases"]:
                name = case["name"]
                reference = torch.load(args.artifacts / "references" / f"{name}.pt", weights_only=True)
                hidden = reference["hidden"]
                product = as_outputs(runtimes["product"](hidden), 1)[0].detach().clone()
                observed = tuple(v.detach().clone() for v in as_outputs(runtimes["observed"](hidden), 4))
                activation = as_outputs(runtimes["activation"](observed[1]), 1)[0].detach().clone()
                values = {"product": (product,), "observed": observed, "activation": (activation,)}
                inputs = {"product": hidden, "observed": hidden, "activation": observed[1]}
                torch.save(values, args.output / f"{name}.pt")
                identities = {
                    "product_reproduces_changed_CI830_product": torch.equal(
                        product, reference["instrumented830_product"]
                    ),
                    "observed_product_matches_probe": torch.equal(observed[0], product),
                    "standalone_activation_matches_observed": torch.equal(activation, observed[3]),
                }
                result["cases"].append({"case": name, "identities": identities})
                comparisons = {"product": (product, reference["cpu_values"][0])}
                comparisons.update(
                    {
                        f"observed_{key}": (a, b)
                        for key, a, b in zip(
                            ("product", "gate", "up", "activation"), observed, reference["cpu_values"], strict=True
                        )
                    }
                )
                comparisons.update(
                    {
                        "activation_on_same_actual_gate": (activation, torch.nn.functional.silu(observed[1])),
                        "observed_activation_on_same_actual_gate": (
                            observed[3],
                            torch.nn.functional.silu(observed[1]),
                        ),
                        "product_on_same_actual_factors": (observed[0], observed[3] * observed[2]),
                    }
                )
                mask = reference["attention_mask"].bool().squeeze(0)
                for feature, (value, expected) in comparisons.items():
                    row = {"case": name, "feature": feature, **metrics(value, expected)}
                    row["valid"] = metrics(value[:, mask], expected[:, mask])
                    row["padding"] = metrics(value[:, ~mask], expected[:, ~mask]) if bool((~mask).any()) else None
                    result["metrics"].append(row)
                    print(json.dumps(row), flush=True)
                for variant, expected in values.items():
                    repeated = as_outputs(runtimes[variant](inputs[variant]), len(expected))
                    result["repeated"][f"{variant}-{name}"] = all(
                        torch.equal(a, b) for a, b in zip(expected, repeated, strict=True)
                    )
                    if variant not in first_inputs:
                        first_inputs[variant], first_outputs[variant] = inputs[variant].clone(), expected
                (args.output / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
            for variant, inputs in first_inputs.items():
                expected = first_outputs[variant]
                returned = as_outputs(runtimes[variant](inputs), len(expected))
                reloaded = rebel.Runtime(args.artifacts / variant / "compiled_model.rbln", tensor_type="pt", device=0)
                fresh = as_outputs(reloaded(inputs), len(expected))
                result["repeated"][f"{variant}-return-reload"] = all(
                    torch.equal(a, b) and torch.equal(a, c) for a, b, c in zip(expected, returned, fresh, strict=True)
                )
                del reloaded
            if not all(result["repeated"].values()):
                raise ValueError("Gating probe state changed values")
            result["changed_graph_reproducer_exact"] = all(
                case["identities"]["product_reproduces_changed_CI830_product"] for case in result["cases"]
            )
            if not result["changed_graph_reproducer_exact"]:
                raise ValueError("Small probe does not exactly reproduce changed CI830 gating; attribution rejected")
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
