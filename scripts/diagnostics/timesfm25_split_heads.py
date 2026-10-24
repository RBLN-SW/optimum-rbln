# Copyright 2026 Rebellions Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by law or agreed in writing, software is distributed AS IS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Diagnostic: return learned TimesFM head branches; add them on CPU in FP32.

ResidualBlock branch ordering follows Transformers 5.15.1 TimesFm2_5ResidualBlock
(Apache-2.0). No learned projection or transformer work is offloaded in the
runtime adapter. CPU learned evaluation is explicitly a diagnostic oracle.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from transformers import TimesFm2_5ModelForPrediction

from . import timesfm25_native_core as native


class LearnedCoreSplitHeads(native.LearnedCore):
    def forward(self, tokenizer_inputs, attention_mask, cos, sin):
        x = self.input_ff_layer(tokenizer_inputs)
        for layer in self.layers:
            x = layer(x, position_embeddings=(cos, sin), attention_mask=attention_mask)
        final = x[:, -1:, :]
        p = final.to(self.point.input_layer.weight.dtype)
        q = final.to(self.quantile.input_layer.weight.dtype)
        point_output = self.point.output_layer(self.point.activation(self.point.input_layer(p)))
        point_residual = self.point.residual_layer(p)
        quantile_output = self.quantile.output_layer(self.quantile.activation(self.quantile.input_layer(q)))
        quantile_residual = self.quantile.residual_layer(q)
        return (
            point_output.reshape(-1, 128, 10),
            point_residual.reshape(-1, 128, 10),
            quantile_output.reshape(-1, 1024, 10),
            quantile_residual.reshape(-1, 1024, 10),
        )


def merge_heads(branches):
    point_output, point_residual, quantile_output, quantile_residual = branches
    return point_output.float() + point_residual.float(), quantile_output.float() + quantile_residual.float()


class SplitHeadAdapter:
    """Actual learned work stays in runtime; only two FP32 tensor adds run here."""

    def __init__(self, runtime, oracle=None, trace_dir=None, baseline_runtime=None):
        self.runtime = runtime
        self.oracle = oracle
        self.trace_dir = trace_dir
        self.baseline_runtime = baseline_runtime
        self.trace_rows = []
        self.context = "raw"
        self.call_index = 0
        self.trace_enabled = True

    def set_context(self, name):
        self.context, self.call_index = name, 0

    def __call__(self, *inputs):
        snapshots = (
            tuple(x.detach().clone() for x in inputs) if self.oracle is not None and self.trace_enabled else None
        )
        start = time.perf_counter()
        branches = tuple(x.detach().clone() for x in self.runtime(*inputs))
        runtime_seconds = time.perf_counter() - start
        actual = merge_heads(branches)
        if snapshots is not None:
            expected_branches = tuple(x.detach().clone() for x in self.oracle(*snapshots))
            expected = merge_heads(expected_branches)
            baseline = (
                tuple(x.detach().clone() for x in self.baseline_runtime(*snapshots))
                if self.baseline_runtime is not None
                else None
            )
            path = self.trace_dir / f"{self.context}-call{self.call_index:02d}.pt"
            torch.save(
                {
                    "inputs": snapshots,
                    "actual_branches": branches,
                    "cpu_branches_same_inputs": expected_branches,
                    "actual_merged": actual,
                    "cpu_merged_same_inputs": expected,
                    "original_baseline_raw": baseline,
                },
                path,
            )
            # Write all branches before metrics; cancellation can hide in merged metrics.
            self.trace_rows.append(
                {
                    "context": self.context,
                    "call": self.call_index,
                    "trace": path.name,
                    "runtime_seconds": runtime_seconds,
                    "branches": [native.metrics(a, e) for a, e in zip(branches, expected_branches, strict=True)],
                    "merged": [native.metrics(a, e) for a, e in zip(actual, expected, strict=True)],
                    "merged_per_field": [
                        [native.metrics(a[..., i], e[..., i]) for i in range(10)]
                        for a, e in zip(actual, expected, strict=True)
                    ],
                    "baseline_merged": None
                    if baseline is None
                    else [native.metrics(a, e) for a, e in zip(baseline, expected, strict=True)],
                    "inputs_unchanged": all(torch.equal(a, b) for a, b in zip(inputs, snapshots, strict=True)),
                }
            )
        self.call_index += 1
        return actual


def native_comparison(actual, data, indices):
    return {
        "quantiles": [native.metrics(a, e) for a, e in zip(actual, data["native_quantiles"][indices], strict=True)],
        "points": [native.metrics(a[..., 5], e) for a, e in zip(actual, data["native_points"][indices], strict=True)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--batches", nargs="+", type=int, default=[1])
    parser.add_argument("--npu", choices=("RBLN-CR13", "RBLN-CA25"), default="RBLN-CR13")
    parser.add_argument("--device", type=int, default=0)
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument(
        "--baseline-artifact", type=Path, help="Original two-output compiled model; B1 and matching NPU only"
    )
    baseline_group.add_argument(
        "--compile-baseline",
        action="store_true",
        help="Compile the original core on the same NPU for a controlled comparison",
    )
    parser.add_argument("--output", type=Path, default=Path("debug-timesfm25-split-heads"))
    args = parser.parse_args()
    assert not (args.cpu_only and (args.baseline_artifact or args.compile_baseline))
    if args.baseline_artifact:
        assert args.batches == [1], "A single original artifact is supported for B1 only"
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    fixtures = Path(__file__).parent / "fixtures" / "timesfm25-native"
    manifest = json.loads((fixtures / "manifest.json").read_text())
    for name, info in manifest.items():
        assert hashlib.sha256((fixtures / name).read_bytes()).hexdigest() == info["sha256"]
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
    split = LearnedCoreSplitHeads(model).eval()
    assert sum(p.numel() for p in split.parameters()) == sum(p.numel() for p in model.parameters()) == 231289280
    report = {
        "purpose": "split-head numerical control, cause unconfirmed until actual NPU comparison",
        "product_accepted": False,
        "cpu_only": args.cpu_only,
        "learned_parameters": 231289280,
        "learned_cpu_offload_in_adapter": False,
        "prior_original_CI794": "FAILED; unchanged baseline retained",
        "original_revision": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
        "converted_revision": "5a9806b9b291fad9233b5249d88263f1846304d3",
        "rows": [],
        "failures": [],
    }
    with torch.no_grad():
        if args.cpu_only:
            example = native.core_inputs(model, torch.randn(1, 508, 32, generator=torch.Generator().manual_seed(42)))
            exported = torch.export.export(split, example, strict=True)
            raw = split(*example)
            raw_exported = exported.module()(*example)
            assert all(torch.equal(a, b) for a, b in zip(raw, raw_exported, strict=True))
            assert all(torch.equal(a, b) for a, b in zip(merge_heads(raw), original(*example), strict=True))
            assert sum(p.numel() for p in exported.parameters()) == 231289280
            report["strict_export"] = {
                "input_shapes": [list(x.shape) for x in example],
                "output_shapes": [list(x.shape) for x in raw],
                "merged_exact_original": True,
                "exported_exact_eager": True,
                "learned_parameters": 231289280,
            }
            torch.save(
                {"inputs": example, "branches": raw, "merged": merge_heads(raw)}, args.output / "cpu-raw-example.pt"
            )
            (args.output / "export-graph.txt").write_text(str(exported.graph))
            trace_dir = args.output / "cpu-raw-traces"
            trace_dir.mkdir(exist_ok=True)
            traced_cpu = SplitHeadAdapter(split, split, trace_dir, original)
            merged = traced_cpu(*example)
            assert all(torch.equal(a, b) for a, b in zip(merged, merge_heads(raw), strict=True))
            assert all(
                metric["max_abs"] == 0
                for key in ("branches", "merged", "baseline_merged")
                for metric in traced_cpu.trace_rows[0][key]
            )
            assert traced_cpu.trace_rows[0]["inputs_unchanged"]
            report["cpu_same_input_adapter_control"] = traced_cpu.trace_rows
            batches = [None]  # CPU proof preserves every fixture's original batch, including B2.
        else:
            version = importlib.metadata.version("rebel-compiler")
            assert version.startswith("0.12.1.dev329+g66a3078e")
            report["compiler"] = version
            report["npu"] = args.npu
            batches = args.batches
        for batch in batches:
            baseline_runtime = None
            baseline_compiled = None
            if batch is None:
                adapter = SplitHeadAdapter(split)
            else:
                assert batch > 0
                inputs = native.core_inputs(
                    model, torch.randn(batch, 508, 32, generator=torch.Generator().manual_seed(42))
                )
                compiled = native.rebel.compile_from_torch(
                    split,
                    input_info=[
                        (name, list(value.shape), "float32")
                        for name, value in zip(
                            ("tokenizer_inputs", "attention_mask", "cos", "sin"), inputs, strict=True
                        )
                    ],
                    example_inputs=list(inputs),
                    npu=args.npu,
                    num_devices=1,
                )
                compiled.save(args.output / f"core-b{batch}.rbln")
                metadata = native.rebel.RBLNCompiledModel.inspect(args.output / f"core-b{batch}.rbln")
                assert [x["name"] for x in metadata["inputs"]] == ["tokenizer_inputs", "attention_mask", "cos", "sin"]
                assert [x["shape"] for x in metadata["outputs"]] == [
                    [batch, 128, 10],
                    [batch, 128, 10],
                    [batch, 1024, 10],
                    [batch, 1024, 10],
                ]
                assert sum(x["device_func_count"] for x in metadata["subgraph"]["stats"]) == 1
                baseline_path = args.baseline_artifact
                if args.compile_baseline:
                    baseline_compiled = native.rebel.compile_from_torch(
                        original,
                        input_info=[
                            (name, list(value.shape), "float32")
                            for name, value in zip(
                                ("tokenizer_inputs", "attention_mask", "cos", "sin"), inputs, strict=True
                            )
                        ],
                        example_inputs=list(inputs),
                        npu=args.npu,
                        num_devices=1,
                    )
                    baseline_path = args.output / f"original-core-b{batch}.rbln"
                    baseline_compiled.save(baseline_path)
                if baseline_path:
                    baseline_metadata = native.rebel.RBLNCompiledModel.inspect(baseline_path)
                    assert baseline_metadata["npu"] == args.npu
                    assert [x["shape"] for x in baseline_metadata["outputs"]] == [[batch, 128, 10], [batch, 1024, 10]]
                    assert sum(x["device_func_count"] for x in baseline_metadata["subgraph"]["stats"]) == 1
                    baseline_runtime = native.rebel.RBLNCompiledModel(baseline_path).create_runtime(
                        tensor_type="pt", device=args.device
                    )
                trace_dir = args.output / f"traces-b{batch}"
                trace_dir.mkdir(exist_ok=True)
                adapter = SplitHeadAdapter(
                    compiled.create_runtime(tensor_type="pt", device=args.device), split, trace_dir, baseline_runtime
                )
            row = {
                "batch": batch,
                "cases": [],
                "baseline_artifact_reused": args.baseline_artifact is not None,
                "original_baseline_control_available": baseline_runtime is not None,
            }
            if batch is not None:
                row["metadata"] = metadata
                row["alloc_per_node"] = compiled.get_alloc_per_node()
                if baseline_runtime is not None:
                    row["original_baseline_metadata"] = baseline_metadata
                row["raw_traces"] = adapter.trace_rows
                cold = adapter(*inputs)
                warm = adapter(*inputs)
                row["cold_runtime_seconds"] = adapter.trace_rows[-2]["runtime_seconds"]
                row["warm_runtime_seconds"] = adapter.trace_rows[-1]["runtime_seconds"]
                row["raw_repeat_equal"] = all(torch.equal(a, b) for a, b in zip(cold, warm, strict=True))
                row["raw_merged"] = [native.metrics(a, e) for a, e in zip(cold, original(*inputs), strict=True)]
                report["failures"].extend(
                    f"raw-b{batch}-{i}" for i, m in enumerate(row["raw_merged"]) if not m["acceptance_pass"]
                )
                if not row["raw_repeat_equal"]:
                    report["failures"].append(f"raw-b{batch}-repeat")
            report["rows"].append(row)
            first_values = first_actual = None
            for name in sorted(manifest, key=lambda x: x == "cpu-constant.pt"):
                data = torch.load(fixtures / name, weights_only=True)
                source = torch.stack(data["inputs"])
                indices = torch.arange(len(source) if batch is None else batch) % len(source)
                values, horizon = source[indices], data["native_points"].shape[1]
                adapter.set_context(name.removesuffix(".pt"))
                adapter.trace_enabled = True
                start = time.perf_counter()
                actual = native.forecast(adapter, model, values, horizon)
                seconds = time.perf_counter() - start
                comparison = native_comparison(actual, data, indices)
                result = {"name": name, "seconds": seconds, **comparison}
                if batch is None:
                    control = native.forecast(original, model, values, horizon)
                    result["cpu_bitwise_original"] = torch.equal(actual, control)
                    assert result["cpu_bitwise_original"], name
                    assert all(m["tight_pass"] for m in comparison["points"] + comparison["quantiles"]), name
                else:
                    adapter.trace_enabled = False
                    repeated = native.forecast(adapter, model, values, horizon)
                    result["repeat_equal"] = torch.equal(actual, repeated)
                    if not result["repeat_equal"]:
                        report["failures"].append(f"{batch}-{name}-repeat")
                    if baseline_runtime is not None:
                        control = native.forecast(baseline_runtime, model, values, horizon)
                        result["original_baseline_control"] = native_comparison(control, data, indices)
                    else:
                        control = None
                torch.save(
                    {
                        "inputs": values,
                        "actual": actual,
                        "native_expected": data["native_quantiles"][indices],
                        "original_control": control,
                    },
                    args.output / f"b{batch}-{name}",
                )
                row["cases"].append(result)
                report["failures"].extend(
                    f"{batch}-{name}-{i}"
                    for i, m in enumerate(comparison["points"] + comparison["quantiles"])
                    if not m["acceptance_pass"]
                )
                if first_values is None:
                    first_values, first_actual = values, actual.clone()
                print("TIMESFM_SPLIT_HEADS " + json.dumps(result), flush=True)
                (args.output / "results.json").write_text(json.dumps(report, indent=2))
            if batch is not None:
                returned = native.forecast(adapter, model, first_values, first_actual.shape[1])
                row["return_equal"] = torch.equal(returned, first_actual)
                adapter.runtime = None
                gc.collect()
                adapter.runtime = native.rebel.RBLNCompiledModel(args.output / f"core-b{batch}.rbln").create_runtime(
                    tensor_type="pt", device=args.device
                )
                reloaded = native.forecast(adapter, model, first_values, first_actual.shape[1])
                row["reload_equal"] = torch.equal(reloaded, first_actual)
                torch.save(
                    {"first": first_actual, "returned": returned, "reloaded": reloaded},
                    args.output / f"reuse-b{batch}.pt",
                )
                if not row["return_equal"] or not row["reload_equal"]:
                    report["failures"].append(f"b{batch}-reuse")
                row["raw_traces"] = adapter.trace_rows
                if not all(trace["inputs_unchanged"] for trace in adapter.trace_rows):
                    report["failures"].append(f"b{batch}-input-mutation")
                adapter.runtime = adapter.baseline_runtime = baseline_runtime = None
                del compiled, baseline_compiled
                gc.collect()
            (args.output / "results.json").write_text(json.dumps(report, indent=2))
    assert not report["failures"], report["failures"]


if __name__ == "__main__":
    main()
