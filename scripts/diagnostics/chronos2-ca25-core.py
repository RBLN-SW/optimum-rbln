"""Compare frozen CA25 artifacts; this diagnostic does not certify model acceptance."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path("/mnt/shared_data/users/jaehunryu/nas_data/model-enable-20261024/chronos-ca25-b8")
OUTPUT = Path("debug-chronos2-core-probe")
ARTIFACTS = {"original": "original-ci747.rbln", "candidate": "host-copy-candidate.rbln"}


def worker(artifact, sync, iterations, label):
    import rebel
    import torch

    torch.set_num_threads(4)
    data = torch.load(ROOT / "failure.pt", weights_only=True)
    reference = torch.load(ROOT / "host-copy-candidate-reference.pt", weights_only=True)
    inputs = tuple(t.clone() for t in data["records"][-1]["inputs"])

    def hashes():
        return [hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in inputs]

    initial_hashes = hashes()
    path = ROOT / ARTIFACTS[artifact]
    compiled = rebel.RBLNCompiledModel(path)
    runtime = compiled.create_runtime(tensor_type="pt", device=0)
    rows, outputs = [], []
    with torch.no_grad():
        for iteration in range(iterations):
            output = runtime(*inputs).clone()
            assert output.shape == reference.shape
            outputs.append(output)
            row = {
                "iteration": iteration,
                "inputs_unchanged": hashes() == initial_hashes,
                "sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest(),
                "delta_first": float((output - outputs[0]).abs().max()),
                "nrmse": float((output - reference).square().mean().sqrt() / reference.square().mean().sqrt()),
            }
            rows.append(row)
            print("CA25_ARTIFACT_COMPARE", artifact, sync, json.dumps(row), flush=True)
    label = label or f"{artifact}-sync{sync}"
    report = {
        "artifact": str(path),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sdk": importlib.metadata.version("rebel-compiler"),
        "force_sync": os.environ["RBLN_RUNTIME_FORCE_SYNC"],
        "host_dump": os.environ.get("RBLN_RUNTIME_HOST_DUMP", "0"),
        "device_dump": os.environ.get("RBLN_RUNTIME_DEVICE_DUMP", "0"),
        "input_hashes": initial_hashes,
        "rows": rows,
    }
    (OUTPUT / f"{label}.json").write_text(json.dumps(report, indent=2))
    torch.save(outputs, OUTPUT / f"{label}.pt")


def dump_probe():
    summary = []
    with tempfile.TemporaryDirectory(prefix="chronos-device-dump-") as temporary:
        for mode in (0, 1):
            directories = []
            for run in (0, 1):
                dump = Path(temporary) / f"mode{mode}-run{run}"
                directories.append(dump)
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--artifact",
                        "original",
                        "--sync",
                        "0",
                        "--iterations",
                        "1",
                        "--label",
                        f"dump{mode}-run{run}",
                    ],
                    env={
                        **os.environ,
                        "RBLN_RUNTIME_FORCE_SYNC": "0",
                        "RBLN_RUNTIME_DEVICE_DUMP": "0",
                        "RBLN_RUNTIME_HOST_DUMP": str(mode * 2),
                        "RBLN_VERBOSE": "trace",
                        "RBLN_RUNTIME_DUMP_PATH": str(dump),
                    },
                    timeout=600,
                    check=True,
                )
            left, right = directories
            files = sorted(
                (p.relative_to(left) for p in left.rglob("*.bin")),
                key=lambda p: (int(p.name.split("_", 1)[0]), str(p)),
            )
            if mode == 1:
                assert files, "Host dump flag produced no binary outputs"
            differences = []
            retained = 0
            for relative in files:
                a, b = left / relative, right / relative
                same = (
                    b.exists() and hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()
                )
                row = {"path": str(relative), "bytes": a.stat().st_size, "equal": same}
                differences.append(row)
                if not same and b.exists() and retained < 4:
                    for name, path in (("left", a), ("right", b)):
                        target = OUTPUT / "first-byte-differences" / name / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, target)
                    retained += 1
            summary.append(
                {
                    "mode": mode,
                    "files": differences,
                    "right_only": [
                        str(p.relative_to(right))
                        for p in right.rglob("*.bin")
                        if not (left / p.relative_to(right)).exists()
                    ],
                }
            )
    report = {
        "scope": "host boundary byte comparison; dtype/layout follow executor; differences require interpretation",
        "runs": summary,
    }
    (OUTPUT / "device-dump-comparison.json").write_text(json.dumps(report, indent=2))
    print("CA25_HOST_DUMP_COMPARISON", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", choices=ARTIFACTS)
    parser.add_argument("--sync", type=int, choices=(0, 1))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--label")
    parser.add_argument("--dump-probe", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    if args.artifact:
        worker(args.artifact, args.sync, args.iterations, args.label)
        return
    if args.dump_probe:
        dump_probe()
        return
    (OUTPUT / "candidate-provenance.json").write_bytes((ROOT / "host-copy-candidate.json").read_bytes())
    for artifact in ARTIFACTS:
        for sync in (0, 1):
            subprocess.run(
                [sys.executable, __file__, "--artifact", artifact, "--sync", str(sync)],
                env={**os.environ, "RBLN_RUNTIME_FORCE_SYNC": str(sync)},
                timeout=600,
                check=True,
            )
    print("CA25_DIAGNOSTIC_COMPLETE; numerical acceptance remains separate", flush=True)


if __name__ == "__main__":
    main()
