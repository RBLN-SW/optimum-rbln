import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import rebel
import torch


BASE = Path("/mnt/shared_data/users/jaehunryu/nas_data/model-enable-20261024")
FIXED = BASE / "chronos-sub64-fix"
OUTPUT = Path("debug-chronos2-core-probe/sub64-fixed-artifacts")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert importlib.metadata.version("rebel-compiler") == "0.12.1.dev319+g8185335c"
    provenance = json.loads((FIXED / "provenance.json").read_text())
    assert provenance["primitive_pin"] == "42d6c1ef6fce4d33c12390c78dcb0abe83c5b94c"
    for name, digest in provenance["artifacts"].items():
        assert hashlib.sha256((FIXED / name).read_bytes()).hexdigest() == digest
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    torch.set_num_threads(4)
    reports = []
    for kind, fixed in (("pv", False), ("pv", True), ("full", False), ("full", True)):
        label = f"{kind}-{'fixed' if fixed else 'original'}"
        if kind == "pv":
            value = torch.randn(576, 12, 8, 64, generator=torch.Generator().manual_seed(42))
            query = torch.eye(8).expand(576, 12, 8, 8).contiguous()
            inputs, reference = (query, value), value
            path = FIXED / "pv576-fixed.rbln" if fixed else BASE / "chronos-pv576/pv576-sdk319.rbln"
        else:
            data = torch.load(BASE / "chronos-ca25-b8/failure.pt", weights_only=True)
            inputs = tuple(t.clone() for t in data["records"][-1]["inputs"])
            reference = torch.load(BASE / "chronos-ca25-b8/host-copy-candidate-reference.pt", weights_only=True)
            path = FIXED / "chronos-b8-fixed.rbln" if fixed else BASE / "chronos-ca25-b8/original-ci747.rbln"
        hashes = [hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in inputs]
        compiled = rebel.RBLNCompiledModel(path)
        runtime = compiled.create_runtime(tensor_type="pt", device=0)
        first = None
        rows = []
        captured = False
        with torch.no_grad():
            for iteration in range(100):
                args = tuple(t.clone() for t in inputs) if iteration % 2 else inputs
                output = runtime(*args).clone()
                assert output.shape == reference.shape
                if first is None:
                    first = output.clone()
                row = {
                    "iteration": iteration,
                    "sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest(),
                    "delta_first": (output - first).abs().max().item(),
                    "nrmse": ((output - reference).square().mean() / reference.square().mean()).sqrt().item(),
                }
                rows.append(row)
                if not captured and (row["delta_first"] or row["nrmse"] > 0.05):
                    torch.save({"output": output, "first": first, "reference": reference}, OUTPUT / f"{label}-bad.pt")
                    captured = True
                if iteration % 10 == 0:
                    print("SUB64_FIXED_COMPARE", label, json.dumps(row), flush=True)
        assert hashes == [hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in inputs]
        report = {
            "label": label,
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "input_hashes": hashes,
            "rows": rows,
        }
        reports.append(report)
        (OUTPUT / "results.json").write_text(json.dumps(reports, indent=2))
        del runtime, compiled, first, output, inputs, reference, args
        gc.collect()
    for report in reports:
        if report["label"].endswith("-fixed"):
            assert len({row["sha256"] for row in report["rows"]}) == 1, report["label"]
            assert all(row["nrmse"] <= 0.05 for row in report["rows"]), report["label"]


if __name__ == "__main__":
    main()
