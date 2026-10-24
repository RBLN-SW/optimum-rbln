"""Observe frozen Chronos mask boundaries without changing runtime inputs or gates."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch


def tensor_digest(value):
    value = value.detach().cpu()
    digest = hashlib.sha256()
    rows = value.reshape(1) if value.ndim == 0 else value
    stride_bytes = max(1, rows[0].numel() * rows.element_size()) if rows.shape[0] else 1
    chunk = max(1, 8 * 1024 * 1024 // stride_bytes)
    for part in rows.split(chunk):
        digest.update(part.contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_summary(value):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": tensor_digest(value),
        "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "contiguous": value.is_contiguous(),
        "data_ptr": value.data_ptr(),
    }


def request_key(inputs):
    values = {
        name: tensor_summary(value)["sha256"] if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def mask_comparison(context_mask, group_ids, actual_time, actual_group, active_batch, chunk=8):
    """Compare logical mask membership; retain raw sentinel differences as evidence."""
    tokens, _, queries, keys = actual_group.shape
    if actual_group.shape[1] != 1 or queries != keys or queries != group_ids.numel():
        raise ValueError("Unexpected full joint group-mask interface")
    attention = torch.cat((context_mask.float(), torch.ones(group_ids.numel(), tokens - context_mask.shape[1])), 1)
    expected_time = (1.0 - attention[:, None, None, :]) * torch.finfo(torch.float32).min
    same = group_ids[:, None] == group_ids[None, :]
    expected_digest = hashlib.sha256()
    result = {
        "actual_time": tensor_summary(actual_time),
        "native_cpu_time": tensor_summary(expected_time),
        "actual_group": tensor_summary(actual_group),
        "native_cpu_group_sha256": None,
        "group_zero_membership_mismatches": 0,
        "active_query_membership_mismatches": 0,
        "blocked_not_negative": 0,
        "expected_permitted_values": 0,
        "actual_permitted_values": 0,
        "actual_negative_values": 0,
        "expected_all_blocked_query_tokens": 0,
        "actual_nan_values": 0,
        "actual_inf_values": 0,
        "actual_positive_values": 0,
        "actual_min": None,
        "actual_max": None,
        "sentinel_note": "IEEE CPU finfo.min and the compiled finite physical sentinel may differ. Zero/blocked membership is compared exactly; no acceptance tolerance is changed.",
    }
    for start in range(0, tokens, chunk):
        permit = same[None, None] & attention[:, start : start + chunk].T[:, None, None, :].bool()
        expected = torch.where(permit, 0.0, torch.finfo(torch.float32).min)
        expected_digest.update(expected.contiguous().numpy().tobytes())
        actual = actual_group[start : start + chunk]
        mismatch = (actual == 0) != permit
        result["group_zero_membership_mismatches"] += int(mismatch.sum())
        result["active_query_membership_mismatches"] += int(mismatch[:, :, :active_batch].sum())
        result["blocked_not_negative"] += int((~permit & ~(actual < 0)).sum())
        result["expected_permitted_values"] += int(permit.sum())
        result["actual_permitted_values"] += int((actual == 0).sum())
        result["actual_negative_values"] += int((actual < 0).sum())
        result["expected_all_blocked_query_tokens"] += int((~permit.any(-1)).sum())
        result["actual_nan_values"] += int(actual.isnan().sum())
        result["actual_inf_values"] += int(actual.isinf().sum())
        result["actual_positive_values"] += int((actual > 0).sum())
        lo, hi = float(actual.min()), float(actual.max())
        result["actual_min"] = lo if result["actual_min"] is None else min(result["actual_min"], lo)
        result["actual_max"] = hi if result["actual_max"] is None else max(result["actual_max"], hi)
    result["native_cpu_group_sha256"] = expected_digest.hexdigest()
    result["time_zero_membership_mismatches"] = int(((actual_time == 0) != (expected_time == 0)).sum())
    result["time_blocked_not_negative"] = int(((expected_time != 0) & ~(actual_time < 0)).sum())
    result["actual_time_nan_values"] = int(actual_time.isnan().sum())
    result["actual_time_inf_values"] = int(actual_time.isinf().sum())
    result["actual_time_min"] = float(actual_time.min())
    result["actual_time_max"] = float(actual_time.max())
    indices = torch.tensor(
        sorted({*range(active_batch), active_batch, min(active_batch + 1, queries - 1), queries - 1})
    )
    indices = indices[indices < queries]
    actual_slice = actual_group.index_select(2, indices).index_select(3, indices).clone()
    selected_permit = (
        same.index_select(0, indices).index_select(1, indices)[None, None]
        & attention.index_select(0, indices).T[:, None, None, :].bool()
    )
    expected_slice = torch.where(selected_permit, 0.0, torch.finfo(torch.float32).min)
    slices = {
        "query_and_key_indices": indices,
        "actual_group_slice": actual_slice,
        "native_cpu_group_slice": expected_slice,
        "actual_time": actual_time.clone(),
        "native_cpu_time": expected_time,
    }
    return result, slices


class MaskDiagnostics:
    """Capture partial-batch evidence before the untouched parity assertions execute."""

    def __init__(self, root, event):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.event = event
        self.pending = {}
        self.active = None
        self.requests = 0

    def _write(self, name, data):
        (self.active["directory"] / name).write_text(json.dumps(data, indent=2) + "\n")

    def reference(self, model, inputs, output):
        if inputs["context"].shape[0] == 3:
            self.pending[request_key(inputs)] = {
                "model": model,
                "expected": output.quantile_preds.detach().cpu().clone(),
            }

    @contextmanager
    def product_request(self, inputs):
        prior = self.active
        self.requests += 1
        self.active = None
        if inputs["context"].shape[0] == 3:
            key = request_key(inputs)
            directory = self.root / f"request-{self.requests:04d}"
            directory.mkdir(exist_ok=False)
            self.active = {"directory": directory, "key": key, "batch": 3, "reference": self.pending.get(key)}
            torch.save(inputs, directory / "original-request.pt")
            self._write(
                "request.json",
                {
                    "key": key,
                    "actual_batch": 3,
                    "inputs": {n: tensor_summary(v) if isinstance(v, torch.Tensor) else v for n, v in inputs.items()},
                },
            )
        try:
            yield
        finally:
            self.active = prior

    def product_output(self, output):
        if self.active is None:
            return
        actual = output.quantile_preds.detach().cpu().clone()
        reference = self.active["reference"]
        torch.save(
            {
                "actual_quantile_preds": actual,
                "original_cpu_expected_quantile_preds": None if reference is None else reference["expected"],
            },
            self.active["directory"] / "forecast-before-original-assert.pt",
        )
        self._write(
            "forecast.json",
            {
                "actual": tensor_summary(actual),
                "expected": None if reference is None else tensor_summary(reference["expected"]),
            },
        )
        self.event(
            {
                "phase": "mask_diagnostics_before_original_assert",
                "directory": str(self.active["directory"]),
                "request_key": self.active["key"],
            }
        )

    def stage(self, phase, call, stage, inputs, output):
        if self.active is None:
            return
        name = stage["name"]
        if phase == "before":
            self._write(f"{name}-inputs.json", {"call": call, "inputs": [tensor_summary(x) for x in inputs]})
            if name == "embedding":
                self.active["normalized"] = inputs
                torch.save(
                    dict(zip(("patched_context", "context_mask", "patched_future", "group_ids"), inputs, strict=True)),
                    self.active["directory"] / "normalized-inputs.pt",
                )
            return
        batch = self.active["batch"]
        if name == "embedding":
            hidden, time_mask, group_mask = output
            normalized = self.active["normalized"]
            mask, groups = normalized[1], normalized[3]
            comparison, slices = mask_comparison(mask, groups, time_mask, group_mask, batch)
            comparison.update(
                {
                    "normalized_group_ids": groups.tolist(),
                    "negative_group_ids": int((groups < 0).sum()),
                    "context_valid_patches": mask.sum(-1).tolist(),
                }
            )
            torch.save(slices, self.active["directory"] / "embedding-mask-slices.pt")
            self._write("embedding-mask-comparison.json", comparison)
            reference = self.active["reference"]
            if reference is not None:
                from scripts.diagnostics.chronos2_encoder_segments import EmbeddingJob

                with torch.no_grad():
                    expected = EmbeddingJob(reference["model"])(*(value[:batch] for value in normalized))[0]
                torch.save(
                    {"actual_active_hidden": hidden[:batch].clone(), "native_cpu_active_hidden": expected},
                    self.active["directory"] / "embedding-active-hidden.pt",
                )
        tensors = tuple(output) if isinstance(output, (tuple, list)) else (output,)
        snapshots = {
            "active_outputs": [x[:batch].clone() for x in tensors[: 1 if name == "embedding" else len(tensors)]]
        }
        if name != "forecast":
            hidden = tensors[0]
            rows = torch.tensor([batch, min(batch + 1, hidden.shape[0] - 1), hidden.shape[0] - 1])
            snapshots["selected_padded_hidden"] = hidden.index_select(0, rows)
            snapshots["selected_padded_rows"] = rows
        torch.save(snapshots, self.active["directory"] / f"{name}-outputs.pt")
        self._write(f"{name}-outputs.json", {"call": call, "outputs": [tensor_summary(x) for x in tensors]})
