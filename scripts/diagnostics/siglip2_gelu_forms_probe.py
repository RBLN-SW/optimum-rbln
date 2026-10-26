"""Compare equivalent tanh GELU forms on original full-sized activation inputs.

This is an operator diagnostic. Neither a green job nor a better component metric
establishes full SigLIP2 acceptance; its original eight-output tests remain required.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


class NativeGelu(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate="tanh")


class SigmoidGelu(torch.nn.Module):
    def forward(self, x):
        inner = math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x)
        return x * torch.sigmoid(2 * inner)


def run(runtime, x):
    output = runtime(x)
    if isinstance(output, (tuple, list)):
        assert len(output) == 1
        output = output[0]
    assert isinstance(output, torch.Tensor)
    return output.clone()


def metrics(actual, expected):
    a, e = actual.double(), expected.double()
    delta = a - e
    return {
        "nrmse": float(delta.square().mean().sqrt() / e.square().mean().sqrt().clamp_min(1e-12)),
        "pcc": float(torch.corrcoef(torch.stack([a.flatten(), e.flatten()]))[0, 1]),
        "max_abs": float(delta.abs().max()),
        "finite": bool(a.isfinite().all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npu", choices=["RBLN-CA25", "RBLN-CR13"])
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    assert args.cpu_only or args.npu is not None
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    manifest = json.loads(Path(__file__).with_name("siglip2_gelu_inputs.json").read_text())
    cases = []
    for row in manifest["files"]:
        path = args.fixture_root / row["filename"]
        assert path.stat().st_size == row["bytes"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
        x = torch.load(path, weights_only=True, map_location="cpu")["npu"][2]
        assert list(x.shape) == [1, 256, 3072]
        assert x.dtype == torch.float32
        assert hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() == row["tensor_sha256"]
        with torch.no_grad():
            expected = NativeGelu()(x)
            equivalent = SigmoidGelu()(x)
        torch.testing.assert_close(equivalent, expected, rtol=1e-5, atol=1e-6)
        cases.append((row["filename"], x, expected, metrics(equivalent, expected)))
    report = {
        "scope": "Existing SigLIP2 activation diagnostic; original model acceptance unchanged",
        "full_model_acceptance": False,
        "fixture_source": manifest["build"],
        "cpu_semantics": [{"fixture": name, **m} for name, _, _, m in cases],
        "cases": [],
    }
    if args.cpu_only:
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print("GELU_CPU_SEMANTICS " + json.dumps(report["cpu_semantics"]), flush=True)
        return

    import rebel

    runtimes = {}
    compiled_models = {}
    for name, model in [("native", NativeGelu()), ("sigmoid", SigmoidGelu())]:
        artifact = args.output / (name + ".rbln")
        compiled = rebel.compile_from_torch(
            model.eval(), input_info=[("x", [1, 256, 3072], "float32")], npu=args.npu, num_devices=1
        )
        compiled.save(str(artifact))
        metadata = rebel.RBLNCompiledModel.inspect(artifact)
        (args.output / (name + "-metadata.json")).write_text(json.dumps(metadata, indent=2) + "\n")
        stats = metadata["subgraph"]["stats"]
        assert stats
        assert all(s["device_func_count"] >= 1 and not s["host_op_reason"] for s in stats)
        compiled_models[name] = compiled
        runtimes[name] = compiled.create_runtime(tensor_type="pt", device=0)
    failures = []
    first = {}
    for index, (fixture, x, expected, _) in enumerate(cases):
        tensors = {"input": x, "reference": expected}
        row = {"fixture": fixture, "npu": args.npu}
        for name, runtime in runtimes.items():
            actual = run(runtime, x)
            repeated = run(runtime, x)
            m = metrics(actual, expected)
            m["repeat_exact"] = torch.equal(actual, repeated)
            if not (m["finite"] and m["nrmse"] <= 0.05 and m["pcc"] >= 0.99 and m["repeat_exact"]):
                failures.append({"fixture": fixture, "form": name, "metrics": m})
            row[name] = m
            tensors[name] = actual
            if index == 0:
                first[name] = actual.clone()
        torch.save(tensors, args.output / (Path(fixture).stem + "-comparison.pt"))
        report["cases"].append(row)
        print("GELU_FORM_CASE " + json.dumps(row), flush=True)
    for name, runtime in runtimes.items():
        assert torch.equal(run(runtime, cases[0][1]), first[name])
    report["component_failures"] = failures
    report["candidate_improves_all_cases"] = all(r["sigmoid"]["nrmse"] < r["native"]["nrmse"] for r in report["cases"])
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    assert not failures, failures
    print("OPERATOR_DIAGNOSTIC_COMPLETE_FULL_MODEL_UNVERIFIED", flush=True)


if __name__ == "__main__":
    main()
