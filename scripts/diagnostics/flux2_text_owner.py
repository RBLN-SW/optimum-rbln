"""Compare native-index Qwen3 outputs; a completed diagnostic is not model acceptance."""

import argparse
import importlib.metadata
import json
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.diagnostics.flux2_transformer_frozen import metrics, require_scheduled_execution, sha256


MANIFEST_SHA = "d135ec04ef29db0746610e0ce4691cef233c8e2446153698d8fd3b5d08b55042"
SDK = "0.12.1.dev333+gf66b88a7"
INDICES = {"baseline": list(range(37)), "selected4": [9, 18, 27, 36], "static_rope": list(range(37))}


def verify_bundle(directory):
    directory = Path(directory)
    if sha256(directory / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Text-owner frozen manifest differs")
    manifest = json.loads((directory / "manifest.json").read_text())
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Frozen text bundle contains missing or unexpected files")
    for name, receipt in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Frozen text files must be ordinary files inside the bundle")
        print(json.dumps({"phase": "verify_file", "file": name, "bytes": receipt["bytes"]}), flush=True)
        if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
            raise ValueError(f"Frozen text checksum differs: {name}")
    expected = {
        "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "sdk": SDK,
        "target": "RBLN-CR13",
        "batch": 1,
        "sequence_length": 512,
        "parameters": 4022468096,
        "decoder_layers": 36,
        "baseline_indices": INDICES["baseline"],
        "selected_indices": INDICES["selected4"],
        "static_indices": INDICES["static_rope"],
        "required_indices": [9, 18, 27],
        "diagnostic_only": True,
        "full_pipeline_npu_acceptance": False,
        "public_selected4_ABI_compatible": False,
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Native full-model text diagnostic contract differs")
    for variant, key in [
        ("baseline", "baseline_binary_sha256"),
        ("selected4", "selected_binary_sha256"),
        ("static_rope", "static_binary_sha256"),
    ]:
        if manifest["files"][f"{variant}/compiled_model.rbln"]["sha256"] != manifest[key]:
            raise ValueError("Text graph identity differs")
    return manifest


def previous_identity(case, values, previous):
    import torch

    if case == "short_component":
        return all(torch.equal(values[i], previous[f"text_layer_{i}"]) for i in (9, 18, 27))
    if case == "full_original_prompt":
        stacked = torch.stack([values[i] for i in (9, 18, 27)], dim=1).permute(0, 2, 1, 3).reshape(1, 512, 7680)
        return torch.equal(stacked, previous["full_prompt_embeddings"])
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("debug-flux2-text-owner"))
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
                raise ValueError("Wrong SDK for frozen text graphs")
            if sha256(Path(ffi_base._LIB._name).resolve()) != manifest["compiler_dso_sha256"]:
                raise ValueError("Loaded SDK library differs")
            if rebel.get_npu_name(0) != "RBLN-CR13":
                raise ValueError("Physical device differs from compiled CR13 target")
            torch.set_num_threads(8)
            torch.set_num_interop_threads(1)
            previous = torch.load(args.artifacts / "previous-actual.pt", weights_only=True)
            result.update(phase="execute", sdk=SDK, rows=[], original_826_identity={}, repeated={})
            saved = {}
            required_pass = {}
            for variant in ("baseline", "selected4", "static_rope"):
                path = args.artifacts / variant / "compiled_model.rbln"
                metadata = rebel.RBLNCompiledModel.inspect(path)
                if metadata["npu"] != "RBLN-CR13" or metadata["num_devices"] != 1:
                    raise ValueError("Text artifact target differs")
                if len(metadata["outputs"]) != len(INDICES[variant]):
                    raise ValueError("Unexpected physical output count")
                if metadata["inputs"] != [
                    {"name": "input_ids", "shape": [1, 512], "dtype": "int64"},
                    {"name": "attention_mask", "shape": [1, 512], "dtype": "int64"},
                ]:
                    raise ValueError("Original native text inputs differ")
                print(json.dumps({"phase": "load_runtime", "variant": variant}), flush=True)
                runtime = rebel.Runtime(path, tensor_type="pt", device=0)
                required_pass[variant] = []
                for case in manifest["cases"]:
                    name = case["name"]
                    reference = torch.load(args.artifacts / "references" / f"{name}.pt", weights_only=True)
                    inputs = reference["inputs"]
                    with torch.inference_mode():
                        outputs = runtime(inputs["input_ids"], inputs["attention_mask"])
                    if not isinstance(outputs, (tuple, list)) or len(outputs) != len(INDICES[variant]):
                        raise ValueError("Runtime text output ABI differs")
                    values = {
                        index: value.detach().clone() for index, value in zip(INDICES[variant], outputs, strict=True)
                    }
                    torch.save(values, args.output / f"{variant}-{name}.pt")
                    saved[(variant, name)] = values
                    if variant == "baseline":
                        identity = previous_identity(name, values, previous)
                        result["original_826_identity"][name] = identity
                        if identity is False:
                            raise ValueError("Original baseline no longer reproduces exact CI826 selected outputs")
                    mask = inputs["attention_mask"].bool().squeeze(0)
                    if int(mask.sum()) != case["valid_tokens"]:
                        raise ValueError("Native prompt-mask identity differs")
                    for index, value in values.items():
                        expected = reference["hidden_states"][index]
                        row = {
                            "variant": variant,
                            "case": name,
                            "native_index": index,
                            "required": index in (9, 18, 27),
                            **metrics(value, expected),
                        }
                        row["valid_tokens"] = metrics(value[:, mask], expected[:, mask])
                        row["padding_tokens"] = (
                            metrics(value[:, ~mask], expected[:, ~mask]) if bool((~mask).any()) else None
                        )
                        result["rows"].append(row)
                        if row["required"]:
                            required_pass[variant].append(row["passed"])
                        print(json.dumps(row), flush=True)
                    repeated = runtime(inputs["input_ids"], inputs["attention_mask"])
                    result["repeated"][f"{variant}-{name}"] = all(
                        torch.equal(value, values[index])
                        for index, value in zip(INDICES[variant], repeated, strict=True)
                    )
                    (args.output / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
                # Reuse the same runtime after different inputs, then reload independently.
                first = torch.load(args.artifacts / "references/short_component.pt", weights_only=True)["inputs"]
                returned = runtime(first["input_ids"], first["attention_mask"])
                reloaded = rebel.Runtime(path, tensor_type="pt", device=0)
                fresh = reloaded(first["input_ids"], first["attention_mask"])
                result["repeated"][f"{variant}-return-first-reload"] = all(
                    torch.equal(a, saved[(variant, "short_component")][index]) and torch.equal(a, b)
                    for index, a, b in zip(INDICES[variant], returned, fresh, strict=True)
                )
                del reloaded, runtime
            result["selected4_vs_baseline_exact"] = {
                case["name"]: {
                    str(i): torch.equal(saved[("selected4", case["name"])][i], saved[("baseline", case["name"])][i])
                    for i in INDICES["selected4"]
                }
                for case in manifest["cases"]
            }
            result["static_rope_vs_baseline_exact"] = {
                case["name"]: {
                    str(i): torch.equal(saved[("static_rope", case["name"])][i], saved[("baseline", case["name"])][i])
                    for i in INDICES["static_rope"]
                }
                for case in manifest["cases"]
            }
            result["required_native_indices_numerically_passed"] = {
                name: all(rows) for name, rows in required_pass.items()
            }
            if not all(result["repeated"].values()):
                raise ValueError("Text runtime repeat/return/reload changed values")
            # This is an A/B experiment with the original BAD graph as a control.
            # Numeric failures remain explicit; success here means evidence was captured.
            result.update(phase="diagnostic_finished", diagnostic_completed=True)
            code = 0
    except BaseException:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        result.update(exit_code=code, seconds=time.monotonic() - started)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        suite = ET.Element("testsuite", name="flux2_text_owner_experiment", tests="1", failures=str(int(code != 0)))
        case = ET.SubElement(suite, "testcase", name=result["phase"])
        if code:
            ET.SubElement(case, "failure").text = result.get("error", "Text diagnostic identity failed")
        ET.ElementTree(suite).write(args.output / "junit.xml", encoding="unicode")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
