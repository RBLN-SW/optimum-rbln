"""Compare frozen CA25 artifacts; this diagnostic does not certify model acceptance."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/mnt/shared_data/users/jaehunryu/nas_data/model-enable-20261024/chronos-ca25-b8")
OUTPUT = Path("debug-chronos2-core-probe")
ARTIFACTS = {"original": "original-ci747.rbln", "candidate": "host-copy-candidate.rbln"}


def worker(artifact, sync):
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
        for iteration in range(20):
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
    label = f"{artifact}-sync{sync}"
    report = {
        "artifact": str(path),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sdk": importlib.metadata.version("rebel-compiler"),
        "force_sync": os.environ["RBLN_RUNTIME_FORCE_SYNC"],
        "input_hashes": initial_hashes,
        "rows": rows,
    }
    (OUTPUT / f"{label}.json").write_text(json.dumps(report, indent=2))
    torch.save(outputs, OUTPUT / f"{label}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", choices=ARTIFACTS)
    parser.add_argument("--sync", type=int, choices=(0, 1))
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    if args.artifact:
        worker(args.artifact, args.sync)
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
