"""Compare frozen full native B1 SigLIP2 GELU semantics on a scheduler-owned CA25.

The SDK329 runtime executes source4b artifacts without recompilation. BEFORE
intentionally retains the old erf semantics. Only AFTER parity determines the
numerical result; failures and tensors from both variants remain on disk.
--verify-only inspects files and interfaces without creating a runtime.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import subprocess
import time
import traceback
from pathlib import Path

import torch


MODEL = "google/siglip2-base-patch16-naflex"
REVISION = "b53b807d3a2d5e2b3911292f2d69e5341cdc064c"
FIXTURE_REVISION = "a9c3851af92cce883828811163f904b61a2550dc"
COMPILER = "4b1449d15e3630fae8b1b86b52dc5d3b90eb85b6"
OPTIMUM = "e861627ace73cc4a466bf99db52b9e84bfadecdb"
FRONTEND_SHA = "522ccfdcaf665165ad9f2e8a4a45a53a9d92d2b4c6a766529aa4545db78e8e1e"
LIBRARY_SHA = "42ee371aaadc50ccb417e1c0c7c4a40deb29cf3dcc27bd5d7f58d0add1a94236"
SDK = "0.12.1.dev329+g66a3078e"
BINARIES = {
    "before": ("46e5abd738de34c26d99240208c22bcc3fb900e1057943e257f4cf41059f1719", 1788866221),
    "after": ("30917e4b97da12c45dc0dade62a70d73678a76b463f5f5aeb326cfecc14d6c62", 1792110915),
}
FILES = {
    "compiled_model.rbln",
    "config.json",
    "rbln_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
INPUTS = [
    ["input_ids", [1, 64], "int64"],
    ["pixel_values", [1, 256, 768], "float32"],
    ["attention_mask", [1, 64], "int64"],
    ["pixel_attention_mask", [1, 256], "int64"],
    ["patch_position_ids", [1, 256], "int64"],
    ["position_ids", [1, 64], "int64"],
]
OUTPUT_SHAPES = [[1, 1], [1, 1], [1, 768], [1, 768], [1, 64, 768], [1, 768], [1, 256, 768], [1, 768]]
NAMES = [
    "image_logits",
    "text_logits",
    "text_embeds",
    "image_embeds",
    "text_hidden",
    "text_pool",
    "vision_hidden",
    "vision_pool",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=True))


def verify_artifacts(root, inspect_model=None):
    """Check source identity, all twelve files, configs, and saved native interfaces."""
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    required = {
        "source_compiler": COMPILER,
        "source_optimum": OPTIMUM,
        "frontend_sha256": FRONTEND_SHA,
        "mapped_librbln_sha256": LIBRARY_SHA,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("Frozen manifest source identity mismatch")
    expected_files = {f"{variant}/{name}" for variant in BINARIES for name in FILES}
    actual_files = {
        str(p.relative_to(root)) for variant in BINARIES for p in (root / variant).rglob("*") if p.is_file()
    }
    if set(manifest["files"]) != expected_files or actual_files != expected_files:
        raise ValueError("Manifest and variant directories must contain exactly the twelve frozen files")
    for name, receipt in manifest["files"].items():
        path = root / name
        if any(p.is_symlink() for p in [path, *path.parents]) or not path.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe artifact path: {name}")
        if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
            raise ValueError(f"Artifact checksum/size mismatch: {name}")
    if inspect_model is None:
        import rebel

        inspect_model = rebel.RBLNCompiledModel.inspect
    metadata = {}
    for variant, (binary_sha, binary_bytes) in BINARIES.items():
        directory = root / variant
        receipt = manifest["files"][f"{variant}/compiled_model.rbln"]
        if (receipt["sha256"], receipt["bytes"]) != (binary_sha, binary_bytes):
            raise ValueError(f"Unexpected frozen binary: {variant}")
        config = json.loads((directory / "rbln_config.json").read_text())
        if any(
            config.get(key) != value
            for key, value in {
                "batch_size": 1,
                "text_batch_size": 1,
                "max_seq_len": 64,
                "max_patches": 256,
                "dtype": "float32",
            }.items()
        ):
            raise ValueError(f"Wrong native capacity/dtype: {variant}")
        configs = config["_compile_cfgs"]
        if len(configs) != 1 or configs[0]["compiled_model_name"] != "compiled_model":
            raise ValueError(f"Expected one full native artifact: {variant}")
        cfg = configs[0]
        if cfg["input_info"] != INPUTS or cfg["npu"] != "RBLN-CA25" or cfg["num_devices"] != 1:
            raise ValueError(f"Wrong compiled interface/target: {variant}")
        model_config = json.loads((directory / "config.json").read_text())
        if model_config["model_type"] != "siglip2" or model_config["vision_config"]["num_patches"] != 256:
            raise ValueError(f"Wrong model checkpoint geometry: {variant}")
        row = inspect_model(directory / "compiled_model.rbln")
        if row["npu"] != "RBLN-CA25" or row["num_devices"] != 1:
            raise ValueError(f"Wrong binary target: {variant}")
        if [[v["name"], v["shape"], v["dtype"]] for v in row["inputs"]] != INPUTS:
            raise ValueError(f"Wrong binary inputs: {variant}")
        if row["outputs"] != [{"shape": s, "dtype": "float32"} for s in OUTPUT_SHAPES]:
            raise ValueError(f"Wrong binary outputs: {variant}")
        stats = row["subgraph"]["stats"]
        if len(stats) != 1 or stats[0]["device_func_count"] != 3:
            raise ValueError(f"Wrong full-model partition count: {variant}")
        metadata[variant] = row
    return {"manifest": manifest, "metadata": metadata}


def flatten(output):
    values = output.to_tuple()
    return tuple(v.detach().cpu().clone() for v in (*values[:4], *values[4], *values[5]))


def tensor_info(value):
    value = value.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
    }


def output_metrics(actual, expected, repeated):
    """Apply unchanged native parity criteria, including scalar-logit handling."""
    if len(actual) != 8 or len(expected) != 8 or len(repeated) != 8:
        raise ValueError("Expected all eight native outputs")
    rows = []
    for name, a, e, r in zip(NAMES, actual, expected, repeated, strict=True):
        row = {"name": name, "actual": tensor_info(a), "expected": tensor_info(e), "repeated": tensor_info(r)}
        checks = {
            "shape": a.shape == e.shape == r.shape,
            "finite": bool(a.isfinite().all()),
            "repeat": torch.equal(a, r),
        }
        if checks["shape"]:
            af, ef = a.float(), e.float()
            row["nrmse"] = float((af - ef).square().mean().sqrt() / ef.square().mean().sqrt().clamp_min(1e-8))
            checks["nrmse"] = row["nrmse"] <= 0.05
            if af.numel() > 1:
                row["pcc"] = float(torch.corrcoef(torch.stack([af.flatten(), ef.flatten()]))[0, 1])
                checks["pcc"] = row["pcc"] >= 0.99
            if name in ("text_embeds", "image_embeds"):
                row["minimum_cosine"] = float(torch.nn.functional.cosine_similarity(af, ef).min())
                checks["minimum_cosine"] = row["minimum_cosine"] >= 0.99
        row["failed_criteria"] = [key for key, passed in checks.items() if not passed]
        rows.append(row)
    return rows


def native_cases(out):
    """Build the original four native image geometries/texts and CPU HF oracle."""
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    reference = AutoModel.from_pretrained(MODEL, revision=REVISION).float().eval()
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    fixture = hf_hub_download(
        "huggingface/documentation-images", "coco_sample.png", repo_type="dataset", revision=FIXTURE_REVISION
    )
    cat = Image.open(fixture).convert("RGB")
    images = [cat, cat.resize((256, 256)), cat.resize((128, 512)), cat.resize((512, 128))]
    texts = ["a photo of cats", "a photo of dogs", "a photo of a train", "a photo of parrots"]
    cases = []
    for index, (image, text) in enumerate(zip(images, texts, strict=True)):
        inputs = dict(
            processor(
                images=[image], text=[text], padding="max_length", max_length=64, truncation=True, return_tensors="pt"
            )
        )
        expected = flatten(reference(**inputs))
        torch.save({"inputs": inputs, "expected": expected}, out / f"cpu-case{index}.pt")
        cases.append((inputs, expected))
    del reference
    gc.collect()
    return cases, {
        "model": MODEL,
        "revision": REVISION,
        "fixture_revision": FIXTURE_REVISION,
        "fixture_sha256": sha256(Path(fixture)),
    }


def capture_variant(variant, artifact, out, cases, loader, events):
    """Preserve every first/repeated/reloaded tensor before evaluating parity."""
    directory = out / variant
    directory.mkdir(parents=True, exist_ok=True)
    report = {"cases": [], "reload_cases": []}
    first_actual = None
    model = None

    def infer(inputs, phase, case, call):
        start = time.perf_counter()
        try:
            return flatten(model(**inputs))
        finally:
            event = {
                "variant": variant,
                "phase": phase,
                "case": case,
                "call": call,
                "host_wall_seconds": time.perf_counter() - start,
            }
            events.append(event)
            dump_json(out / "timings.json", events)

    try:
        for phase in ("initial", "reload"):
            start = time.perf_counter()
            model = loader(artifact)
            events.append({"variant": variant, "phase": phase, "load_host_wall_seconds": time.perf_counter() - start})
            for index, (inputs, expected) in enumerate(cases):
                actual = infer(inputs, phase, index, "cold" if index == 0 else "warm")
                repeated = infer(inputs, phase, index, "warm-repeat")
                # This payload is durable before any criteria are evaluated.
                torch.save(
                    {"inputs": inputs, "expected": expected, "actual": actual, "repeated": repeated},
                    directory / f"{phase}-case{index}.pt",
                )
                row = {
                    "case": index,
                    "inputs": {key: tensor_info(value) for key, value in inputs.items()},
                    "outputs": output_metrics(actual, expected, repeated),
                }
                if phase == "initial":
                    if index == 0:
                        first_actual = actual
                    report["cases"].append(row)
                else:
                    original = torch.load(
                        directory / f"initial-case{index}.pt", map_location="cpu", weights_only=True
                    )["actual"]
                    row["reload_equal"] = [torch.equal(a, b) for a, b in zip(actual, original, strict=True)]
                    report["reload_cases"].append(row)
                dump_json(directory / "results.json", report)
            returned = infer(cases[0][0], phase, 0, "warm-return-to-first")
            torch.save({"actual": returned, "first_actual": first_actual}, directory / f"{phase}-return-to-first.pt")
            report[f"{phase}_return_to_first_equal"] = [
                torch.equal(a, b) for a, b in zip(returned, first_actual, strict=True)
            ]
            del model
            model = None
            gc.collect()
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if model is not None:
            del model
        gc.collect()
        dump_json(directory / "results.json", report)
    return report


def failures(report, check_parity=True):
    result = []
    for phase in ("cases", "reload_cases"):
        if len(report.get(phase, [])) != 4:
            result.append(f"{phase}: incomplete")
        for case in report.get(phase, []):
            result.extend(
                f"{phase}/{case['case']}/{row['name']}: {row['failed_criteria']}"
                for row in case["outputs"]
                if row["failed_criteria"]
                and (check_parity or set(row["failed_criteria"]) & {"shape", "finite", "repeat"})
            )
            if phase == "reload_cases" and not all(case["reload_equal"]):
                result.append(f"reload_cases/{case['case']}: changed outputs")
    for phase in ("initial", "reload"):
        equal = report.get(f"{phase}_return_to_first_equal", [])
        if len(equal) != 8 or not all(equal):
            result.append(f"{phase}: return-to-first changed outputs")
    if "error" in report:
        result.append(report["error"])
    return result


def run_comparison(artifacts, out, cases, loader):
    """Complete both variants before deciding AFTER acceptance."""
    events, reports = [], {}
    for variant in BINARIES:
        reports[variant] = capture_variant(variant, artifacts / variant, out, cases, loader, events)
    result = {
        "variants": reports,
        "before_failed_criteria": failures(reports["before"]),
        "after_failed_criteria": failures(reports["after"]),
        "before_stability_failures": failures(reports["before"], check_parity=False),
        "acceptance_scope": "frozen source4b native B1 A/B only; not SDK330 or packaged product acceptance",
    }
    # Baseline parity failures are expected; execution failures still invalidate A/B.
    result["passed"] = not result["after_failed_criteria"] and not result["before_stability_failures"]
    dump_json(out / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("debug-siglip2-gelu-frozen"))
    parser.add_argument("--device", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only and args.device is None:
        parser.error("--device must be supplied by the scheduler")
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = verify_artifacts(args.artifacts)
    # Inspection uses the frozen binary's embedded compiler label; source identity
    # comes from the checked manifest, since these source builds retain a stale label.
    receipt["runtime_sdk"] = importlib.metadata.version("rebel-compiler")
    if receipt["runtime_sdk"] != SDK:
        raise ValueError(f"Expected SDK runtime {SDK}")
    receipt["validation_checkout"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    receipt["validation_diff_sha256"] = hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest()
    receipt["packages"] = {
        name: importlib.metadata.version(name) for name in ("torch", "transformers", "optimum-rbln", "rebel-compiler")
    }
    receipt["partition_interpretation"] = (
        "Three device functions in each variant; constant-table gathers, masks, rsqrt and transfers on host. host_op_reason labels alone do not identify learned host compute."
    )
    dump_json(args.out / "verification.json", receipt)
    if args.verify_only:
        print(
            "SIGLIP2_GELU_VERIFIED " + json.dumps({"runtime_created": False, "artifacts": str(args.artifacts)}),
            flush=True,
        )
        return
    from optimum.rbln import RBLNSiglip2Model

    torch.set_num_threads(4)
    with torch.no_grad():
        cases, fixture = native_cases(args.out)
        dump_json(args.out / "fixture.json", fixture)
        result = run_comparison(
            args.artifacts,
            args.out,
            cases,
            lambda directory: RBLNSiglip2Model.from_pretrained(directory, rbln_device=args.device),
        )
    print(
        "SIGLIP2_GELU_COMPLETE "
        + json.dumps(
            {
                "passed": result["passed"],
                "before_failed_criteria": result["before_failed_criteria"],
                "after_failed_criteria": result["after_failed_criteria"],
                "scope": result["acceptance_scope"],
            }
        ),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
