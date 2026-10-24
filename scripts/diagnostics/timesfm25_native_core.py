# Copyright 2025 Google LLC and 2026 Rebellions Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by law or agreed in writing, software is distributed AS IS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Scheduler-only native TimesFM2.5 full-weight numerical diagnostics.

Running-statistics/RevIN utilities derive from Google TimesFM Apache-2.0,
commit e51928e27119cb17bebc005be2696b75e0a9e688. Native goldens are pinned
original-checkpoint forecasts; all learned computation executes on the NPU.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import rebel
import torch
from torch import nn
from transformers import TimesFm2_5ModelForPrediction


_TOLERANCE = 1e-6


def update_running_stats(
    n: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Updates the running stats."""
    is_legit = torch.logical_not(mask)
    inc_n = torch.sum(is_legit.to(x.dtype), dim=-1)

    inc_mu_numerator = torch.sum(x * is_legit, dim=-1)
    inc_n_safe = torch.where(inc_n == 0, 1.0, inc_n)
    inc_mu = inc_mu_numerator / inc_n_safe
    inc_mu = torch.where(inc_n == 0, 0.0, inc_mu)

    inc_var_numerator = torch.sum(((x - inc_mu.unsqueeze(-1)) ** 2) * is_legit, dim=-1)
    inc_var = inc_var_numerator / inc_n_safe
    inc_var = torch.where(inc_n == 0, 0.0, inc_var)
    inc_sigma = torch.sqrt(inc_var)

    new_n = n + inc_n
    new_n_safe = torch.where(new_n == 0, 1.0, new_n)

    new_mu = (n * mu + inc_mu * inc_n) / new_n_safe
    new_mu = torch.where(new_n == 0, 0.0, new_mu)

    term1 = n * sigma.pow(2)
    term2 = inc_n * inc_sigma.pow(2)
    term3 = n * (mu - new_mu).pow(2)
    term4 = inc_n * (inc_mu - new_mu).pow(2)

    new_var = (term1 + term2 + term3 + term4) / new_n_safe
    new_var = torch.where(new_n == 0, 0.0, new_var)
    new_sigma = torch.sqrt(torch.clamp(new_var, min=0.0))

    return (w := (new_n, new_mu, new_sigma), w)


def revin(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    reverse: bool = False,
):
    """Reversible instance normalization."""
    if len(mu.shape) == len(x.shape) - 1:
        mu = mu[..., None]
        sigma = sigma[..., None]
    elif len(mu.shape) == len(x.shape) - 2:
        mu = mu[..., None, None]
        sigma = sigma[..., None, None]

    if reverse:
        return x * sigma + mu
    else:
        return (x - mu) / torch.where(sigma < _TOLERANCE, 1.0, sigma)


class LearnedCore(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.input_ff_layer = model.model.input_ff_layer
        self.layers = model.model.layers
        self.point = model.output_projection_point
        self.quantile = model.output_projection_quantiles

    def forward(self, tokenizer_inputs, attention_mask, cos, sin):
        x = self.input_ff_layer(tokenizer_inputs)
        for layer in self.layers:
            x = layer(x, position_embeddings=(cos, sin), attention_mask=attention_mask)
        final = x[:, -1:, :]
        return self.point(final).reshape(-1, 128, 10), self.quantile(final).reshape(-1, 1024, 10)


def core_inputs(model, normalized_patches):
    b, p, _ = normalized_patches.shape
    if p > 512:
        raise ValueError("Native context budget exceeded")
    padding = 512 - p
    masked = torch.arange(512)[None, :] < padding
    masked = masked.expand(b, -1)
    values = torch.nn.functional.pad(normalized_patches, (0, 0, padding, 0))
    tokens = torch.cat((values, masked[:, :, None].expand(-1, -1, 32).float()), -1)
    causal = torch.arange(512)[:, None] >= torch.arange(512)[None, :]
    allowed = causal[None, :, :] & ~masked[:, None, :]
    attention = torch.where(allowed[:, None], 0.0, torch.finfo(torch.float32).min)
    positions = torch.arange(512)[None, :].expand(b, -1) - padding
    cos, sin = model.model.rotary_emb(values, positions)
    return tokens, attention, cos, sin


def decode(core, model, inputs, horizon):
    b = inputs.shape[0]
    count = torch.zeros(b)
    mean = torch.zeros(b)
    std = torch.zeros(b)
    normalized = []

    def append(values):
        nonlocal count, mean, std
        for patch in values.reshape(b, -1, 32).unbind(1):
            (count, mean, std), _ = update_running_stats(
                count, mean, std, patch, torch.zeros_like(patch, dtype=torch.bool)
            )
            normalized.append(revin(patch, mean, std))

    append(inputs)
    outputs = []
    quantiles = None
    for step in range((horizon + 127) // 128):
        args = core_inputs(model, torch.stack(normalized, 1))
        point, spread = core(*args)
        point = revin(point, mean, std, reverse=True)
        if step == 0:
            quantiles = revin(spread, mean, std, reverse=True)
        outputs.append(point)
        if step < (horizon - 1) // 128:
            append(point[:, :, 5])
    return torch.cat(outputs, 1)[:, :horizon], quantiles[:, :horizon]


def forecast(core, model, values, horizon):
    if values.shape[-1] + horizon > 16384:
        raise ValueError("Native context plus horizon exceeds16384")
    positive = (values >= 0).all(-1)
    mu = values.mean(-1, keepdim=True)
    sigma = values.std(-1, keepdim=True)
    normalized = revin(values, mu, sigma)
    point, quantile = decode(core, model, normalized, horizon)
    flip_point, flip_quantile = decode(core, model, -normalized, horizon)

    def flip(x):
        return torch.cat((x[..., :1], x[..., 1:].flip(-1)), -1)

    point = (point - flip(flip_point)) / 2
    quantile = (quantile - flip(flip_quantile)) / 2
    for i in (1, 2, 3, 4, 6, 7, 8, 9):
        point[:, :, i] = quantile[:, :, i] - quantile[:, :, 5] + point[:, :, 5]
    result = revin(point, mu, sigma, reverse=True)
    return torch.where(positive[:, None, None], result.clamp_min(0), result)


def metrics(actual, expected):
    delta = actual - expected
    constant = bool((expected.max() - expected.min()) == 0)
    nrmse = float(delta.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8))
    pcc = None if constant else float(torch.corrcoef(torch.stack((actual.flatten(), expected.flatten())))[0, 1])
    maximum = float(delta.abs().max())
    return {
        "nrmse": nrmse,
        "pcc": pcc,
        "max_abs": maximum,
        "finite": bool(actual.isfinite().all()),
        "tight_pass": maximum <= 1e-4 and (constant or nrmse <= 1e-4),
        "acceptance_pass": bool(actual.isfinite().all())
        and (maximum <= 1e-6 if constant else nrmse <= 0.05 and pcc >= 0.99),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    fixtures = Path(__file__).parent / "fixtures" / "timesfm25-native"
    out = Path("debug-timesfm25")
    out.mkdir(exist_ok=True)
    compiler_version = importlib.metadata.version("rebel-compiler")
    assert compiler_version.startswith("0.12.1.dev329+g66a3078e")
    manifest = json.loads((fixtures / "manifest.json").read_text())
    for name, info in manifest.items():
        assert hashlib.sha256((fixtures / name).read_bytes()).hexdigest() == info["sha256"]
    model = (
        TimesFm2_5ModelForPrediction.from_pretrained(
            "google/timesfm-2.5-200m-transformers",
            revision="5a9806b9b291fad9233b5249d88263f1846304d3",
            attn_implementation="eager",
        )
        .float()
        .eval()
    )
    core = LearnedCore(model).eval()
    assert sum(p.numel() for p in core.parameters()) == sum(p.numel() for p in model.parameters())
    report = {
        "compiler": compiler_version,
        "original_revision": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
        "converted_revision": "5a9806b9b291fad9233b5249d88263f1846304d3",
        "rows": [],
    }
    failures = []
    with torch.no_grad():
        for batch in args.batches:
            assert batch > 0
            inputs = core_inputs(model, torch.randn(batch, 508, 32, generator=torch.Generator().manual_seed(42)))
            expected_raw = core(*inputs)
            compiled = rebel.compile_from_torch(
                core,
                input_info=[
                    (name, list(value.shape), "float32")
                    for name, value in zip(("tokenizer_inputs", "attention_mask", "cos", "sin"), inputs, strict=True)
                ],
                example_inputs=list(inputs),
                npu="RBLN-CR13",
                num_devices=1,
            )
            compiled.save(out / f"core-b{batch}.rbln")
            metadata = rebel.RBLNCompiledModel.inspect(out / f"core-b{batch}.rbln")
            assert [entry["name"] for entry in metadata["inputs"]] == [
                "tokenizer_inputs",
                "attention_mask",
                "cos",
                "sin",
            ]
            assert sum(entry["device_func_count"] for entry in metadata["subgraph"]["stats"]) == 1
            runtime = compiled.create_runtime(tensor_type="pt", device=args.device)
            actual_raw = tuple(x.clone() for x in runtime(*inputs))
            torch.save({"inputs": inputs, "actual": actual_raw, "expected": expected_raw}, out / f"raw-b{batch}.pt")
            raw_metrics = [metrics(a, e) for a, e in zip(actual_raw, expected_raw, strict=True)]
            failures.extend(
                f"raw-b{batch}-{i}" for i, metric in enumerate(raw_metrics) if not metric["acceptance_pass"]
            )
            row = {
                "batch": batch,
                "alloc_per_node": compiled.get_alloc_per_node(),
                "raw": raw_metrics,
                "metadata": metadata,
                "cases": [],
            }
            report["rows"].append(row)
            first_values = first_actual = None
            for name in sorted(manifest, key=lambda value: value == "cpu-constant.pt"):
                data = torch.load(fixtures / name, map_location="cpu", weights_only=True)
                source_values = torch.stack(data["inputs"])
                indices = torch.arange(batch) % len(source_values)
                values = source_values[indices]
                horizon = data["native_points"].shape[1]
                expected = data["native_quantiles"][indices]
                start = time.perf_counter()
                actual = forecast(runtime, model, values, horizon)
                seconds = time.perf_counter() - start
                repeated = forecast(runtime, model, values, horizon)
                torch.save(
                    {"inputs": values, "actual": actual, "expected": expected, "repeated": repeated},
                    out / f"b{batch}-{name}",
                )
                per_series = [metrics(a, e) for a, e in zip(actual, expected, strict=True)]
                points = [metrics(a[:, 5], e) for a, e in zip(actual, data["native_points"][indices], strict=True)]
                result = {
                    "name": name,
                    "horizon": horizon,
                    "seconds": seconds,
                    "per_series": per_series,
                    "points": points,
                    "repeat_equal": torch.equal(actual, repeated),
                }
                row["cases"].append(result)
                failures.extend(
                    f"b{batch}-{name}-{i}"
                    for i, metric in enumerate(per_series + points)
                    if not metric["acceptance_pass"]
                )
                if not result["repeat_equal"]:
                    failures.append(f"b{batch}-{name}-repeat")
                print("TIMESFM_NATIVE " + json.dumps(result), flush=True)
                (out / "results.json").write_text(json.dumps(report, indent=2))
                if first_values is None:
                    first_values, first_actual = values, actual.clone()
            returned = forecast(runtime, model, first_values, first_actual.shape[1])
            row["return_equal"] = torch.equal(returned, first_actual)
            del runtime
            runtime = rebel.RBLNCompiledModel(out / f"core-b{batch}.rbln").create_runtime(
                tensor_type="pt", device=args.device
            )
            reloaded = forecast(runtime, model, first_values, first_actual.shape[1])
            row["reload_equal"] = torch.equal(reloaded, first_actual)
            torch.save({"returned": returned, "reloaded": reloaded, "first": first_actual}, out / f"reuse-b{batch}.pt")
            if not row["return_equal"] or not row["reload_equal"]:
                failures.append(f"b{batch}-reuse")
            del runtime, compiled
            gc.collect()
            report["failures"] = failures
            (out / "results.json").write_text(json.dumps(report, indent=2))
    assert not failures, failures


if __name__ == "__main__":
    main()
