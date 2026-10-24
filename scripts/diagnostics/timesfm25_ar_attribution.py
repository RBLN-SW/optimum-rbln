# Copyright 2026 Rebellions Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by law or agreed in writing, software is distributed AS IS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Attribute native TimesFM numerical drift; teacher forcing is an oracle control.

Both diagnostic paths aggregate actual runtime outputs. Only teacher-forced
feedback comes from the CPU oracle. This is never a product acceptance path.
The original free-running diagnostic and its thresholds remain unchanged.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from transformers import TimesFm2_5ModelForPrediction

from . import timesfm25_native_core as native


def comparison(actual, expected):
    """Descriptive errors without attaching acceptance to oracle controls."""
    return {key: value for key, value in native.metrics(actual, expected).items() if not key.endswith("pass")}


def raw_comparison(actual, expected):
    return {
        "all": comparison(actual, expected),
        "per_field": [comparison(actual[..., i], expected[..., i]) for i in range(10)],
        "per_128_positions": [
            comparison(actual[:, start : start + 128], expected[:, start : start + 128])
            for start in range(0, actual.shape[1], 128)
        ],
    }


def traced_decode(runtime, oracle, model, inputs, horizon, mode, sign, out, steps, timings):
    """Keep the full normalized prefix; vary only the source of AR feedback."""
    b = inputs.shape[0]
    count, mean, std = (torch.zeros(b) for _ in range(3))
    normalized = []

    def append(values):
        nonlocal count, mean, std
        for patch in values.reshape(b, -1, 32).unbind(1):
            (count, mean, std), _ = native.update_running_stats(
                count, mean, std, patch, torch.zeros_like(patch, dtype=torch.bool)
            )
            normalized.append(native.revin(patch, mean, std))

    append(inputs)
    outputs = []
    quantiles = None
    for step in range((horizon + 127) // 128):
        args = native.core_inputs(model, torch.stack(normalized, 1))
        assert [list(x.shape) for x in args] == [[1, 512, 64], [1, 1, 512, 512], [1, 512, 80], [1, 512, 80]]
        sent_inputs = tuple(x.detach().clone() for x in args)
        start = time.perf_counter()
        actual = tuple(x.detach().clone() for x in runtime(*args))
        runtime_seconds = time.perf_counter() - start
        timings.append(runtime_seconds)
        start = time.perf_counter()
        expected = tuple(x.detach().clone() for x in oracle(*sent_inputs))
        oracle_seconds = time.perf_counter() - start
        # Persist the exact same runtime input and both raw outputs before any
        # metrics, RevIN or postprocessing. Runtime buffers cannot alias a later call.
        path = out / f"{mode}-{sign}-step{step:02d}.pt"
        torch.save(
            {
                "input_names": ["tokenizer_inputs", "attention_mask", "cos", "sin"],
                "inputs": sent_inputs,
                "actual_raw": actual,
                "cpu_raw_same_inputs": expected,
                "count": count.clone(),
                "mean": mean.clone(),
                "std": std.clone(),
            },
            path,
        )
        actual_point = native.revin(actual[0], mean, std, reverse=True)
        cpu_point = native.revin(expected[0], mean, std, reverse=True)
        if step == 0:
            quantiles = native.revin(actual[1], mean, std, reverse=True)
        outputs.append(actual_point)
        entry = {
            "mode": mode,
            "sign": sign,
            "step": step,
            "trace": path.name,
            "prefix_patches": len(normalized),
            "leading_masked_patches": 512 - len(normalized),
            "last_valid_token_index": 511,
            "last_valid_position_id": len(normalized) - 1,
            "last_valid_token_min": float(args[0][:, -1, :32].min()),
            "last_valid_token_max": float(args[0][:, -1, :32].max()),
            "runtime_seconds": runtime_seconds,
            "cpu_oracle_seconds": oracle_seconds,
            "runtime_inputs_unchanged": all(torch.equal(a, b) for a, b in zip(args, sent_inputs, strict=True)),
            "point_raw_same_inputs": raw_comparison(actual[0], expected[0]),
            "spread_raw_same_inputs": raw_comparison(actual[1], expected[1]),
            "feedback_median_same_inputs": comparison(actual_point[:, :, 5], cpu_point[:, :, 5]),
        }
        steps.append(entry)
        if step < (horizon - 1) // 128:
            append((cpu_point if mode == "teacher_forced" else actual_point)[:, :, 5])
    return torch.cat(outputs, 1)[:, :horizon], quantiles[:, :horizon]


def traced_forecast(runtime, oracle, model, values, horizon, mode, out, steps, timings):
    if values.shape[-1] + horizon > 16384:
        raise ValueError("Native context plus horizon exceeds16384")
    positive = (values >= 0).all(-1)
    mu = values.mean(-1, keepdim=True)
    sigma = values.std(-1, keepdim=True)
    normalized = native.revin(values, mu, sigma)
    point, quantile = traced_decode(runtime, oracle, model, normalized, horizon, mode, "positive", out, steps, timings)
    flip_point, flip_quantile = traced_decode(
        runtime, oracle, model, -normalized, horizon, mode, "negative", out, steps, timings
    )

    def flip(x):
        return torch.cat((x[..., :1], x[..., 1:].flip(-1)), -1)

    point = (point - flip(flip_point)) / 2
    quantile = (quantile - flip(flip_quantile)) / 2
    for i in (1, 2, 3, 4, 6, 7, 8, 9):
        point[:, :, i] = quantile[:, :, i] - quantile[:, :, 5] + point[:, :, 5]
    result = native.revin(point, mu, sigma, reverse=True)
    return torch.where(positive[:, None, None], result.clamp_min(0), result)


def compare_prefixes(out, horizon):
    """Separate accumulated prefix/state drift from same-input raw core error."""
    comparisons = []
    for sign in ("positive", "negative"):
        for step in range((horizon + 127) // 128):
            free = torch.load(out / f"free_running-{sign}-step{step:02d}.pt", weights_only=True)
            teacher = torch.load(out / f"teacher_forced-{sign}-step{step:02d}.pt", weights_only=True)
            comparisons.append(
                {
                    "sign": sign,
                    "step": step,
                    "tokenizer_mask_equal": torch.equal(free["inputs"][0][..., 32:], teacher["inputs"][0][..., 32:]),
                    "attention_mask_equal": torch.equal(free["inputs"][1], teacher["inputs"][1]),
                    "cos_equal": torch.equal(free["inputs"][2], teacher["inputs"][2]),
                    "sin_equal": torch.equal(free["inputs"][3], teacher["inputs"][3]),
                    "normalized_prefix": comparison(free["inputs"][0][..., :32], teacher["inputs"][0][..., :32]),
                    "last_four_valid_patches": comparison(
                        free["inputs"][0][:, -4:, :32], teacher["inputs"][0][:, -4:, :32]
                    ),
                    "running_mean_max_abs": float((free["mean"] - teacher["mean"]).abs().max()),
                    "running_std_max_abs": float((free["std"] - teacher["std"]).abs().max()),
                    "cpu_point_raw_prefix_drift": raw_comparison(
                        free["cpu_raw_same_inputs"][0], teacher["cpu_raw_same_inputs"][0]
                    ),
                    "actual_point_raw_prefix_drift": raw_comparison(free["actual_raw"][0], teacher["actual_raw"][0]),
                    "cpu_median_denormalized_prefix_drift": comparison(
                        native.revin(free["cpu_raw_same_inputs"][0], free["mean"], free["std"], reverse=True)[:, :, 5],
                        native.revin(teacher["cpu_raw_same_inputs"][0], teacher["mean"], teacher["std"], reverse=True)[
                            :, :, 5
                        ],
                    ),
                }
            )
    return comparisons


def run_case(runtime, oracle, model, data, out, cpu_only, timings):
    out.mkdir(exist_ok=True)
    # This focused B1 control includes the failing positive row of the native B2 fixture.
    values = torch.stack(data["inputs"][:1])
    horizon = data["native_points"].shape[1]
    expected = data["native_quantiles"][:1]
    start = time.perf_counter()
    original = native.forecast(runtime, model, values, horizon)
    original_seconds = time.perf_counter() - start
    steps = []
    free = traced_forecast(runtime, oracle, model, values, horizon, "free_running", out, steps, timings)
    teacher = traced_forecast(runtime, oracle, model, values, horizon, "teacher_forced", out, steps, timings)
    torch.save(
        {
            "inputs": values,
            "native_expected": expected,
            "original_free_running": original,
            "traced_free_running": free,
            "teacher_forced_actual_outputs": teacher,
        },
        out / "forecasts.pt",
    )
    free_metrics = native.metrics(original, expected)
    point_metrics = native.metrics(original[:, :, 5], data["native_points"][:1])
    row = {
        "horizon": horizon,
        "original_free_running_seconds": original_seconds,
        "baseline_quantiles": free_metrics,
        "baseline_point": point_metrics,
        "traced_free_equals_original": torch.equal(free, original),
        "teacher_forced_oracle_control_only": comparison(teacher, expected),
        "teacher_point_per_128": [
            comparison(teacher[:, start : start + 128, 5], expected[:, start : start + 128, 5])
            for start in range(0, horizon, 128)
        ],
        "free_point_per_128": [
            comparison(free[:, start : start + 128, 5], expected[:, start : start + 128, 5])
            for start in range(0, horizon, 128)
        ],
        "steps": steps,
        "paired_prefixes": compare_prefixes(out, horizon),
    }
    failures = []
    if not free_metrics["acceptance_pass"] or not point_metrics["acceptance_pass"]:
        failures.append("original-free-running-native-acceptance")
    if not row["traced_free_equals_original"]:
        failures.append("traced-free-running-changed-baseline")
    if cpu_only:
        row["cpu_teacher_native_tight"] = native.metrics(teacher, expected)["tight_pass"]
        row["cpu_teacher_free_tight"] = native.metrics(teacher, original)["tight_pass"]
        if not all((row["cpu_teacher_native_tight"], row["cpu_teacher_free_tight"], free_metrics["tight_pass"])):
            failures.append("cpu-oracle-native-tight-control")
    (out / "results.json").write_text(json.dumps(row, indent=2))
    print(
        "TIMESFM_AR_ATTRIBUTION "
        + json.dumps({k: v for k, v in row.items() if k not in ("steps", "paired_prefixes")}),
        flush=True,
    )
    return row, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu-only", action="store_true", help="Test orchestration with CPU as both runtime and oracle"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("debug-timesfm25-ar"))
    args = parser.parse_args()
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
    core = native.LearnedCore(model).eval()
    parameter_count = sum(p.numel() for p in core.parameters())
    assert parameter_count == sum(p.numel() for p in model.parameters()) == 231289280
    report = {
        "purpose": "attribution-only; CPU oracle feedback is not a product path",
        "product_accepted": False,
        "cpu_only": args.cpu_only,
        "learned_parameters": parameter_count,
        "cases": {},
        "failures": [],
        "original_revision": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
        "converted_revision": "5a9806b9b291fad9233b5249d88263f1846304d3",
    }
    with torch.no_grad():
        inputs = native.core_inputs(model, torch.randn(1, 508, 32, generator=torch.Generator().manual_seed(42)))
        runtime = core
        if not args.cpu_only:
            compiler_version = importlib.metadata.version("rebel-compiler")
            assert compiler_version.startswith("0.12.1.dev329+g66a3078e")
            report["compiler"] = compiler_version
            compiled = native.rebel.compile_from_torch(
                core,
                input_info=[
                    (name, list(value.shape), "float32")
                    for name, value in zip(("tokenizer_inputs", "attention_mask", "cos", "sin"), inputs, strict=True)
                ],
                example_inputs=list(inputs),
                npu="RBLN-CR13",
                num_devices=1,
            )
            compiled.save(args.output / "core-b1.rbln")
            metadata = native.rebel.RBLNCompiledModel.inspect(args.output / "core-b1.rbln")
            assert [entry["name"] for entry in metadata["inputs"]] == [
                "tokenizer_inputs",
                "attention_mask",
                "cos",
                "sin",
            ]
            assert sum(entry["device_func_count"] for entry in metadata["subgraph"]["stats"]) == 1
            report["metadata"] = metadata
            report["alloc_per_node"] = compiled.get_alloc_per_node()
            runtime = compiled.create_runtime(tensor_type="pt", device=args.device)
        # Preserve the original same-input raw control. Label cold and warmed calls;
        # these include host invocation/output copies and are not device-only timing.
        start = time.perf_counter()
        cold = tuple(x.clone() for x in runtime(*inputs))
        report["cold_runtime_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        warm = tuple(x.clone() for x in runtime(*inputs))
        report["warm_runtime_seconds"] = time.perf_counter() - start
        raw_expected = tuple(core(*inputs))
        torch.save(
            {"inputs": inputs, "actual": cold, "expected": raw_expected, "warm": warm}, args.output / "raw-b1.pt"
        )
        report["raw_control"] = [native.metrics(a, e) for a, e in zip(cold, raw_expected, strict=True)]
        report["raw_repeat_equal"] = all(torch.equal(a, b) for a, b in zip(cold, warm, strict=True))
        report["failures"].extend(
            f"raw-{i}" for i, metric in enumerate(report["raw_control"]) if not metric["acceptance_pass"]
        )
        if not report["raw_repeat_equal"]:
            report["failures"].append("raw-repeat")
        timings = []
        for name in ("cpu-native_limit_horizon1024.pt", "cpu-mixed_positive_falling.pt"):
            data = torch.load(fixtures / name, weights_only=True, map_location="cpu")
            row, failures = run_case(
                runtime, core, model, data, args.output / name.removesuffix(".pt"), args.cpu_only, timings
            )
            report["cases"][name] = row
            report["failures"].extend(f"{name}:{failure}" for failure in failures)
            (args.output / "results.json").write_text(json.dumps(report, indent=2))
        report["traced_runtime_seconds"] = timings
        (args.output / "results.json").write_text(json.dumps(report, indent=2))
    # Good teacher-forced results can never clear a failing free-running baseline.
    assert not report["failures"], report["failures"]


if __name__ == "__main__":
    main()
