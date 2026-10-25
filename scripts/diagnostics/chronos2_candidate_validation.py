"""Run the original full-model B512 suite with the reviewed compiler fix candidate."""

import argparse
import json
import os
import shutil
from pathlib import Path

from scripts.diagnostics.chronos2_frozen_validation import (
    NAMES,
    run_suite,
    sha256,
    validate_frozen,
    verify_runtime_library,
)


CANDIDATE = "8f59b810c9a226c26225c0480d1b06ae30f1d4cf"
BASELINE = "0eeb40e41cae94010206d8426646e8347bf76d07873663eee5df7cd66bae93c5"
OVERLAYS = {
    "extend_scaled_dot_product_attention": "4966a4b175b74d505f2747f55c20b6ae187ba87b1b79cdd14f3d68d4e6a1317d",
    "scaled_dot_product_attention": "8bd9e8fca067f39b584cd474ffac0abcd391945b921e0c3ac5a4d022f755bb93",
}


def require_candidate_source(record):
    if record.get("source_candidate") != CANDIDATE:
        raise ValueError("Unreviewed candidate source")
    rows = record.get("candidate_source_overlay", [])
    if len(rows) != 2 or {row["function"]: row["sha256"] for row in rows} != OVERLAYS:
        raise ValueError("Candidate lowering functions differ from the reviewed patch")


def validate_candidate(root, digest, inspect_model=None):
    root = Path(root)
    if sha256(root / "manifest.json") != digest:
        raise ValueError("Candidate manifest SHA mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    require_candidate_source(manifest)
    if manifest.get("kind") != "existing-Chronos-full-model-compiler-fix-candidate":
        raise ValueError("Expected the explicit whole-model compiler candidate scope")
    if manifest.get("full_model_acceptance") is not False:
        raise ValueError("An unexecuted bundle cannot claim model acceptance")
    if sha256(root / "baseline-manifest.json") != BASELINE:
        raise ValueError("Original frozen baseline changed")
    baseline = json.loads((root / "baseline-manifest.json").read_text())
    for name, row in manifest["files"].items():
        path = root / name
        if Path(name).name != name or path.is_symlink():
            raise ValueError("Candidate files must stay inside the frozen root")
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("Candidate file bytes changed: " + name)
    expected_files = set(manifest["files"]) | {"manifest.json"}
    if {p.name for p in root.iterdir()} != expected_files:
        raise ValueError("Missing or extra candidate bundle files")
    for name in [
        "embedding.rbln",
        "forecast.rbln",
        "config.json",
        "original_rbln_config.json",
        "compile_configs.json",
        "original-test-generate-inputs.pt",
    ]:
        if manifest["files"][name]["sha256"] != baseline["files"][name]["sha256"]:
            raise ValueError("An unaffected stage/config/input changed: " + name)
    partition = json.loads((root / "original-parameter-partition.json").read_text())
    if (
        sha256(root / "original-parameter-partition.json")
        != "ed33571405739acc2d0b20e412bbde287be6467465cae0dc701f2e588fc02b2c"
    ):
        raise ValueError("Original trained parameter partition changed")
    all_weights = None
    for stage in manifest["stages"][1:-1]:
        report = json.loads((root / ("compile-" + stage["name"] + ".json")).read_text())
        require_candidate_source(report)
        if report["phase"] != "compiled" or not report["all_checkpoint_weight_tensors_exact"] or report["guard_calls"]:
            raise ValueError("Unverified compile or trained weights")
        if report["sha256"] != stage["sha256"] or report["metadata"] != stage["metadata"]:
            raise ValueError("Compile report/artifact binding changed")
        actual = {w["native_name"]: (w["shape"], w["elements"]) for w in report["stage_weights"]}
        expected = {w["name"]: (w["shape"], w["elements"]) for w in partition[stage["name"]]["parameters"]}
        if actual != expected or len(actual) != 39:
            raise ValueError("Original contiguous three-block parameter ownership changed")
        if report["placement"]["arithmetic_host"] or report["placement"]["device_functions"] != 1:
            raise ValueError("Learned arithmetic must remain on device")
        if all_weights is not None and all_weights != report["checkpoint_weights"]:
            raise ValueError("Checkpoint weights differ between encoder stages")
        all_weights = report["checkpoint_weights"]
        if any(w["sha256"] != all_weights[w["native_name"]]["sha256"] for w in report["stage_weights"]):
            raise ValueError("Stage weights differ from the original checkpoint")
    # Keep the original strict geometry, interfaces and existing-stack checks.
    validate_frozen(root, digest, inspect_model)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stage-to", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.artifacts = args.artifacts.resolve()
    manifest = validate_candidate(args.artifacts, args.manifest_sha256)
    if args.stage_to is not None:
        destination = args.stage_to
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            validate_candidate(destination, args.manifest_sha256)
        else:
            temporary = destination.with_name(destination.name + f".partial-{os.getpid()}")
            if temporary.exists():
                raise ValueError("Preserving an existing partial staging directory")
            shutil.copytree(args.artifacts, temporary)
            validate_candidate(temporary, args.manifest_sha256)
            temporary.rename(destination)
        print("CANDIDATE_STAGED_VERIFIED; no NPU APIs", flush=True)
        return
    if args.verify_only:
        print("CANDIDATE_VERIFY_ONLY_PASS; no compiler or NPU imports", flush=True)
        return
    if not os.environ.get("BUILDKITE_JOB_ID") or os.environ.get("RBLN_TEST_NPU") != "RBLN-CA25":
        raise ValueError("Runtime requires the scheduler-owned CA25 job")
    if any(
        name in os.environ
        for name in ["SAVE_ARTIFACTS_PATH", "REUSE_ARTIFACTS_PATH", "RBLN_DUMMY_DEVICE", "TVM_LIBRARY_PATH"]
    ):
        raise ValueError("Artifact compatibility modes and runtime-library overrides are forbidden")
    if args.out is None:
        raise ValueError("Runtime execution needs an output directory")
    if sha256(Path(__file__)) != manifest["candidate_runner_sha256"]:
        raise ValueError("Candidate runner source mismatch")
    provenance = verify_runtime_library("4e1bc153f001ae5a8694478052c7df4b05d8b5dbc09c6a27c06153e6decc7a9e")
    import rebel

    validate_candidate(args.artifacts, args.manifest_sha256, rebel.RBLNCompiledModel.inspect)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    events = args.out / "events.jsonl"

    def event(value):
        with events.open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print("CHRONOS_CANDIDATE " + json.dumps(value), flush=True)

    event({"phase": "pre_runtime_verified", "runtime": provenance, "candidate": CANDIDATE, "stages": list(NAMES)})
    os.environ["OPTIMUM_RBLN_TEST_LEVEL"] = "full"
    os.chdir(args.out)
    run_suite(args.artifacts.resolve(), manifest, args.out, event)


if __name__ == "__main__":
    main()
