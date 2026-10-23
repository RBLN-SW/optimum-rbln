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
            for relative in files:
                a, b = left / relative, right / relative
                same = (
                    b.exists() and hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()
                )
                row = {"path": str(relative), "bytes": a.stat().st_size, "equal": same}
                differences.append(row)
            output_differences = [r for r in differences if not r["equal"] and "[output_" in r["path"]]
            selected = {Path(r["path"]) for r in output_differences[:4]}
            if output_differences:
                first_node = output_differences[0]["path"].split("_", 1)[0]
                selected.update(p for p in files if p.name.split("_", 1)[0] == first_node)
            for relative in sorted(selected):
                for name, directory in (("left", left), ("right", right)):
                    path = directory / relative
                    if path.exists():
                        target = OUTPUT / "first-output-differences" / name / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, target)
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


def matmul_probe():
    import gc

    import rebel
    import torch

    class Scores(torch.nn.Module):
        def forward(self, query, key):
            return torch.matmul(query, key.transpose(-1, -2))

    torch.set_num_threads(4)
    model = Scores().eval()
    reports = []
    for sequence, batch in ((577, 8), (576, 8), (577, 16), (577, 64)):
        generator = torch.Generator().manual_seed(42)
        query, key = (torch.randn(sequence, 12, batch, 64, generator=generator) for _ in range(2))
        name = f"matmul-s{sequence}-b{batch}"
        with torch.no_grad():
            reference = model(query, key)
            compiled = rebel.compile_from_torch(
                model,
                input_info=[("query", list(query.shape), "float32"), ("key", list(key.shape), "float32")],
                npu="RBLN-CA25",
                num_devices=1,
            )
            compiled.save(str(OUTPUT / f"{name}.rbln"))
            runtime = compiled.create_runtime(tensor_type="pt", device=0)
            first = None
            rows = []
            for iteration in range(10):
                output = runtime(query, key).clone()
                if first is None:
                    first = output.clone()
                row = {
                    "iteration": iteration,
                    "sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest(),
                    "delta_first": float((output - first).abs().max()),
                    "nrmse": float((output - reference).square().mean().sqrt() / reference.square().mean().sqrt()),
                }
                rows.append(row)
                print("CA25_SYNTHETIC_MATMUL", name, json.dumps(row), flush=True)
            reports.append({"name": name, "input_shape": list(query.shape), "rows": rows})
            if any(r["delta_first"] for r in rows):
                torch.save(
                    {"query": query, "key": key, "first": first, "last": output, "reference": reference},
                    OUTPUT / f"{name}-failure.pt",
                )
        del runtime, compiled, query, key, reference, output, first
        gc.collect()
    (OUTPUT / "synthetic-matmul.json").write_text(json.dumps(reports, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", choices=ARTIFACTS)
    parser.add_argument("--sync", type=int, choices=(0, 1))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--label")
    parser.add_argument("--dump-probe", action="store_true")
    parser.add_argument("--matmul-probe", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    if args.artifact:
        worker(args.artifact, args.sync, args.iterations, args.label)
        return
    if args.matmul_probe:
        matmul_probe()
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
