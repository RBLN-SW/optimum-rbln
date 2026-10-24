import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import rebel
import torch
from transformers import Dinov2Config, Dinov2Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    out = Path("debug-dinov2-static-jobs")
    out.mkdir(exist_ok=True)
    report = {"phase": "artifact verification", "compiler": importlib.metadata.version("rebel-compiler"), "runs": []}
    try:
        if not args.verify_only:
            assert report["compiler"] == "0.12.1.dev328+g970308c6"
        manifest = json.loads((args.artifacts / "provenance.json").read_text())
        assert manifest["source_head"] == "970308c6eec6e0b794d9a20313d1dc77bc050837"
        report["provenance"] = manifest
        for name, entry in manifest["files"].items():
            path = args.artifacts / name
            assert path.stat().st_size == entry["bytes"]
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
        names = ["embedding", "group", "tail"]
        metadata = [rebel.RBLNCompiledModel.inspect(args.artifacts / f"{name}.rbln") for name in names]
        assert len({item["uuid"] for item in metadata}) == 1
        report["artifact_uuids"] = [item["uuid"] for item in metadata]
        fixture = torch.load(args.artifacts / "cpu-reference.pt", weights_only=True)
        reference = Dinov2Model(Dinov2Config.from_dict(fixture["config"])).float().eval()
        reference.load_state_dict(fixture["model_state_dict"], strict=True)
        torch.set_num_threads(4)
        with torch.no_grad():
            cpu = reference(fixture["input"], return_dict=False)
            for a, e in zip(cpu, fixture["expected"], strict=True):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
        if args.verify_only:
            report["phase"] = "artifact hashes and CPU fixture verified; NPU not executed"
            return
        saved = None
        with torch.no_grad():
            for reload_index in range(2):
                report["phase"] = f"runtime reload {reload_index}"
                compiled = [rebel.RBLNCompiledModel(args.artifacts / f"{name}.rbln") for name in names]
                runtimes = [model.create_runtime(device=args.device, tensor_type="pt") for model in compiled]
                for index, values in enumerate([fixture["input"], fixture["input"] * 0.5, fixture["input"]]):
                    expected = reference(values, return_dict=False)
                    start = time.perf_counter()
                    runtimes[0](values)
                    embedded = time.perf_counter()
                    runtimes[1](torch.zeros(1, dtype=torch.int16))
                    grouped = time.perf_counter()
                    actual = tuple(value.clone() for value in runtimes[2](torch.ones(1, dtype=torch.float32)))
                    finished = time.perf_counter()
                    row = {
                        "reload": reload_index,
                        "request": index,
                        "stage_ms": [
                            (embedded - start) * 1000,
                            (grouped - embedded) * 1000,
                            (finished - grouped) * 1000,
                        ],
                        "outputs": [],
                    }
                    report["runs"].append(row)
                    for a, e in zip(actual, expected, strict=True):
                        row["outputs"].append(
                            {
                                "shape": list(a.shape),
                                "finite": bool(a.isfinite().all()),
                                "pcc": float(torch.corrcoef(torch.stack([a.flatten(), e.flatten()]))[0, 1]),
                                "nrmse": float((a - e).square().mean().sqrt() / e.square().mean().sqrt()),
                            }
                        )
                    torch.save(
                        {"input": values, "expected": expected, "actual": actual},
                        out / f"reload{reload_index}-request{index}.pt",
                    )
                    (out / "results.json").write_text(json.dumps(report, indent=2))
                    for metric in row["outputs"]:
                        assert metric["finite"], metric
                        assert metric["pcc"] >= 0.99, metric
                        assert metric["nrmse"] <= 0.05, metric
                    if index == 0 and reload_index == 0:
                        saved = actual
                    if index in (0, 2):
                        for a, e in zip(actual, saved, strict=True):
                            torch.testing.assert_close(a, e, rtol=0, atol=0)
                    print("DINOV2_STATIC_JOBS " + json.dumps(row), flush=True)
                del runtimes, compiled
                gc.collect()
        report["phase"] = "passed tiny interface only"
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        (out / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
