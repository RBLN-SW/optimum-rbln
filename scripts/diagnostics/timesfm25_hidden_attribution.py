# Copyright 2026 Rebellions Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by law or agreed in writing, software is distributed AS IS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Attribute original TimesFM raw errors across the final hidden/head boundary.

All learned work in the actual path remains on NPU. CPU heads on identical
NPU hidden states are oracle evaluations only. This probe does not replace
the original native forecast acceptance suite or establish product acceptance.
"""

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path

import torch
from transformers import TimesFm2_5ModelForPrediction

from . import timesfm25_native_core as native


class LearnedCoreWithHidden(native.LearnedCore):
    def forward(self, tokenizer_inputs, attention_mask, cos, sin):
        x = self.input_ff_layer(tokenizer_inputs)
        for layer in self.layers:
            x = layer(x, position_embeddings=(cos, sin), attention_mask=attention_mask)
        final = x[:, -1:, :]
        return self.point(final).reshape(-1, 128, 10), self.quantile(final).reshape(-1, 1024, 10), final


def heads_on_hidden(core, hidden):
    """CPU oracle only; never supplies actual runtime or autoregressive outputs."""
    return core.point(hidden).reshape(-1, 128, 10), core.quantile(hidden).reshape(-1, 1024, 10)


def comparison(actual, expected):
    return {key: value for key, value in native.metrics(actual, expected).items() if not key.endswith("pass")}


def component_errors(actual, heads_on_actual_hidden, reference):
    total = actual - reference
    head = actual - heads_on_actual_hidden
    trunk = heads_on_actual_hidden - reference
    recomposed = head + trunk
    denominator = total.square().sum().clamp_min(1e-30)
    return {
        "total": comparison(actual, reference),
        "head_same_hidden": comparison(actual, heads_on_actual_hidden),
        "trunk_propagated_through_cpu_head": comparison(heads_on_actual_hidden, reference),
        "head_error_rmse": float(head.square().mean().sqrt()),
        "trunk_error_rmse": float(trunk.square().mean().sqrt()),
        "head_dot_total_fraction": float((head * total).sum() / denominator),
        "trunk_dot_total_fraction": float((trunk * total).sum() / denominator),
        "decomposition_max_abs": float((total - recomposed).abs().max()),
        "decomposition_close": torch.allclose(total, recomposed, atol=1e-6, rtol=1e-5),
        "per_field": [
            {
                "total": comparison(actual[..., i], reference[..., i]),
                "head_same_hidden": comparison(actual[..., i], heads_on_actual_hidden[..., i]),
                "trunk_propagated": comparison(heads_on_actual_hidden[..., i], reference[..., i]),
            }
            for i in range(10)
        ],
    }


def load_inputs():
    root = Path(__file__).parent / "fixtures" / "timesfm25-hidden"
    manifest = json.loads((root / "manifest.json").read_text())
    records = []
    for name, info in manifest.items():
        compressed = (root / name).read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == info["sha256"]
        data = torch.load(io.BytesIO(gzip.decompress(compressed)), weights_only=True)
        assert [list(x.shape) for x in data["inputs"]] == [[1, 512, 64], [1, 1, 512, 512], [1, 512, 80], [1, 512, 80]]
        assert [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in data["inputs"]] == info["input_sha256"]
        records.append((name.removesuffix(".pt.gz"), data, info))
    assert len(records) == 4
    return records


def inspect_model(path, output_shapes, npu):
    metadata = native.rebel.RBLNCompiledModel.inspect(path)
    assert metadata["npu"] == npu
    assert [x["name"] for x in metadata["inputs"]] == ["tokenizer_inputs", "attention_mask", "cos", "sin"]
    assert [x["shape"] for x in metadata["outputs"]] == output_shapes
    assert sum(x["device_func_count"] for x in metadata["subgraph"]["stats"]) == 1
    return metadata


def compile_core(core, inputs, path, npu):
    compiled = native.rebel.compile_from_torch(
        core,
        input_info=[
            (name, list(value.shape), "float32")
            for name, value in zip(("tokenizer_inputs", "attention_mask", "cos", "sin"), inputs, strict=True)
        ],
        example_inputs=list(inputs),
        npu=npu,
        num_devices=1,
    )
    compiled.save(path)
    return compiled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--npu", choices=("RBLN-CR13", "RBLN-CA25"), default="RBLN-CR13")
    parser.add_argument("--device", type=int, default=0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--baseline-artifact", type=Path)
    group.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("debug-timesfm25-hidden"))
    args = parser.parse_args()
    assert args.cpu_only or args.baseline_artifact or args.compile_baseline, (
        "A same-device original baseline is required"
    )
    assert not (args.cpu_only and (args.baseline_artifact or args.compile_baseline))
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    records = load_inputs()
    model = (
        TimesFm2_5ModelForPrediction.from_pretrained(
            "google/timesfm-2.5-200m-transformers",
            revision="5a9806b9b291fad9233b5249d88263f1846304d3",
            attn_implementation="eager",
            local_files_only=args.cpu_only,
        )
        .float()
        .eval()
    )
    original = native.LearnedCore(model).eval()
    core = LearnedCoreWithHidden(model).eval()
    assert sum(p.numel() for p in core.parameters()) == sum(p.numel() for p in model.parameters()) == 231289280
    report = {
        "purpose": "hidden/head interface attribution only",
        "product_accepted": False,
        "native_acceptance_exercised": False,
        "cpu_only": args.cpu_only,
        "learned_parameters": 231289280,
        "learned_cpu_offload_in_actual_path": False,
        "baseline_failures_retained": "CI794 original and CI797 split-head native gates remain failed",
        "converted_revision": "5a9806b9b291fad9233b5249d88263f1846304d3",
        "cases": [],
        "failures": [],
        "interpretation": "Error components are vectors; RMS magnitudes do not add. Signed dot fractions are not independent variance percentages.",
    }
    with torch.no_grad():
        if args.cpu_only:
            exported = torch.export.export(core, records[0][1]["inputs"], strict=True)
            assert sum(p.numel() for p in exported.parameters()) == 231289280
            runtime = exported.module()
            baseline_runtime = original
            (args.output / "export-graph.txt").write_text(str(exported.graph))
            report["strict_export"] = {
                "parameter_count": 231289280,
                "input_shapes": [list(x.shape) for x in records[0][1]["inputs"]],
                "output_shapes": [[1, 128, 10], [1, 1024, 10], [1, 1, 1280]],
            }
        else:
            version = importlib.metadata.version("rebel-compiler")
            assert version.startswith("0.12.1.dev329+g66a3078e")
            report["compiler"] = version
            report["npu"] = args.npu
            compiled = compile_core(core, records[0][1]["inputs"], args.output / "core-hidden.rbln", args.npu)
            report["metadata"] = inspect_model(
                args.output / "core-hidden.rbln", [[1, 128, 10], [1, 1024, 10], [1, 1, 1280]], args.npu
            )
            runtime = compiled.create_runtime(tensor_type="pt", device=args.device)
            baseline_path = args.baseline_artifact
            if args.compile_baseline:
                baseline_path = args.output / "original-core.rbln"
                baseline_compiled = compile_core(original, records[0][1]["inputs"], baseline_path, args.npu)
                del baseline_compiled
            report["baseline_metadata"] = inspect_model(baseline_path, [[1, 128, 10], [1, 1024, 10]], args.npu)
            baseline_runtime = native.rebel.RBLNCompiledModel(baseline_path).create_runtime(
                tensor_type="pt", device=args.device
            )
        for name, saved, provenance in records:
            inputs = tuple(x.clone() for x in saved["inputs"])
            actual = tuple(x.detach().clone() for x in runtime(*inputs))
            baseline = tuple(x.detach().clone() for x in baseline_runtime(*inputs))
            expected = tuple(x.detach().clone() for x in core(*saved["inputs"]))
            oracle_heads = tuple(x.detach().clone() for x in heads_on_hidden(core, actual[2]))
            # All learned oracle outputs stay separate from the actual outputs.
            # Preserve exact inputs and raw evidence before calculating any metrics.
            torch.save(
                {
                    "inputs": saved["inputs"],
                    "actual": actual,
                    "original_baseline": baseline,
                    "cpu_reference": expected,
                    "cpu_heads_on_actual_hidden": oracle_heads,
                    "saved_ci794_original_raw": saved["ci794_original_raw"],
                    "saved_ci794_cpu_raw": saved["ci794_cpu_raw"],
                },
                args.output / f"{name}.pt",
            )
            exact_original = [torch.equal(a, b) for a, b in zip(actual[:2], baseline, strict=True)]
            parts = [component_errors(a, h, e) for a, h, e in zip(actual[:2], oracle_heads, expected[:2], strict=True)]
            row = {
                "name": name,
                "provenance": provenance,
                "instrumented_raw_exact_original_baseline": exact_original,
                "instrumented_vs_original_baseline": [
                    comparison(a, b) for a, b in zip(actual[:2], baseline, strict=True)
                ],
                "original_baseline_vs_cpu": [comparison(a, e) for a, e in zip(baseline, expected[:2], strict=True)],
                "original_baseline_exact_saved_ci794": [
                    torch.equal(a, b) for a, b in zip(baseline, saved["ci794_original_raw"], strict=True)
                ],
                "hidden": comparison(actual[2], expected[2]),
                "point": parts[0],
                "spread": parts[1],
                "inputs_unchanged": all(torch.equal(a, b) for a, b in zip(inputs, saved["inputs"], strict=True)),
                "cpu_reference_vs_saved_ci794": [
                    comparison(a, e) for a, e in zip(expected[:2], saved["ci794_cpu_raw"], strict=True)
                ],
            }
            if not all(exact_original):
                report["failures"].append(name + ":output-instrumentation-changed-baseline")
            if not row["inputs_unchanged"] or not all(x["decomposition_close"] for x in parts):
                report["failures"].append(name + ":input-or-decomposition-control")
            if args.cpu_only:
                eager = core(*saved["inputs"])
                row["exported_exact_eager"] = all(torch.equal(a, b) for a, b in zip(actual, eager, strict=True))
                row["cpu_oracle_same_hidden_exact"] = all(
                    torch.equal(a, b) for a, b in zip(actual[:2], oracle_heads, strict=True)
                )
                assert row["exported_exact_eager"], name
                assert row["cpu_oracle_same_hidden_exact"], name
                assert all(
                    m["tight_pass"]
                    for m in [native.metrics(a, e) for a, e in zip(expected[:2], saved["ci794_cpu_raw"], strict=True)]
                ), name
            report["cases"].append(row)
            print(
                "TIMESFM_HIDDEN "
                + json.dumps(
                    {
                        "name": name,
                        "raw_exact_original": exact_original,
                        "hidden": row["hidden"],
                        "point_total": parts[0]["total"],
                        "head_rmse": parts[0]["head_error_rmse"],
                        "trunk_rmse": parts[0]["trunk_error_rmse"],
                    }
                ),
                flush=True,
            )
            (args.output / "results.json").write_text(json.dumps(report, indent=2))
    # Changed heads are a confounded instrument, even when their metrics improve.
    assert not report["failures"], report["failures"]


if __name__ == "__main__":
    main()
