"""Retain matched original CPU/NPU predictions and per-series acceptance metrics."""

import hashlib
import json
import weakref
from contextlib import contextmanager
from pathlib import Path


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def input_identity(inputs):
    import torch

    record = {}
    for name, value in sorted(inputs.items()):
        record[name] = (
            {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha(value)}
            if isinstance(value, torch.Tensor)
            else value
        )
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(), record


def series_metrics(actual, expected):
    import torch

    if actual.shape != expected.shape or actual.ndim != 3:
        raise ValueError("Original batched quantile prediction interface changed")
    actual, expected = actual.double().flatten(1), expected.double().flatten(1)
    finite = torch.isfinite(actual).all(1) & torch.isfinite(expected).all(1)
    rows = []
    for index, (got, reference) in enumerate(zip(actual, expected, strict=True)):
        if not finite[index]:
            rows.append({"series": index, "finite": False, "nrmse": None, "pcc": None, "passed": False})
            continue
        nrmse = ((got - reference).square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-8)).item()
        left, right = got - got.mean(), reference - reference.mean()
        denominator = (left.norm() * right.norm()).item()
        pcc = torch.dot(left, right).item() / denominator if denominator else None
        passed = nrmse <= 0.05 and (pcc >= 0.99 if pcc is not None else torch.equal(got, reference))
        rows.append({"series": index, "finite": True, "nrmse": nrmse, "pcc": pcc, "passed": passed})
    return rows


class PredictionAudit:
    def __init__(self, root, checkpoint_weights):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.weights = checkpoint_weights
        self.checked_models = weakref.WeakSet()
        self.checked_model_count = 0
        self.references = {}
        self.first_outputs = {}
        self.reference_calls = []
        self.calls = []
        self.active = None
        self.files = {}

    def _save_tensor(self, value):
        import torch

        digest = tensor_sha(value)
        name = "tensor-" + digest + ".pt"
        if name not in self.files:
            torch.save({"prediction": value.detach().cpu().clone()}, self.root / name)
            self.files[name] = {"raw_sha256": digest, "shape": list(value.shape), "dtype": str(value.dtype)}
        return name, digest

    def reference(self, model, inputs, output):
        import torch

        if model not in self.checked_models:
            state = model.state_dict()
            if set(state) != set(self.weights):
                raise ValueError("CPU oracle checkpoint tensor membership changed")
            for name, value in state.items():
                if (
                    list(value.shape) != self.weights[name]["shape"]
                    or tensor_sha(value) != self.weights[name]["sha256"]
                ):
                    raise ValueError("CPU oracle trained weights changed: " + name)
            self.checked_models.add(model)
            self.checked_model_count += 1
        key, description = input_identity(inputs)
        filename, digest = self._save_tensor(output.quantile_preds)
        inputs_name = "inputs-" + key + ".pt"
        if inputs_name not in self.files:
            torch.save(inputs, self.root / inputs_name)
            self.files[inputs_name] = {"identity": key, "inputs": description}
        self.references[key] = {
            "prediction": output.quantile_preds.detach().cpu().clone(),
            "file": filename,
            "raw_sha256": digest,
            "inputs_file": inputs_name,
        }
        self.reference_calls.append(
            {"input_identity": key, "prediction_file": filename, "prediction_raw_sha256": digest}
        )

    @contextmanager
    def product_request(self, inputs):
        if self.active is not None:
            raise ValueError("Unexpected nested product request")
        key, description = input_identity(inputs)
        self.active = {"input_identity": key, "inputs": description}
        try:
            yield
        finally:
            after, _ = input_identity(inputs)
            if after != key:
                raise ValueError("Product modified original request inputs")
            self.active = None

    def product_output(self, output):
        if self.active is None:
            raise ValueError("Prediction escaped its original input binding")
        row = dict(self.active)
        key = row["input_identity"]
        reference = self.references.get(key)
        if reference is None:
            row["scope"] = "original generation method without a CPU parity oracle"
        else:
            filename, digest = self._save_tensor(output.quantile_preds)
            metrics = series_metrics(output.quantile_preds, reference["prediction"])
            first = self.first_outputs.setdefault(key, digest)
            row.update(
                prediction_file=filename,
                prediction_raw_sha256=digest,
                reference_file=reference["file"],
                reference_raw_sha256=reference["raw_sha256"],
                inputs_file=reference["inputs_file"],
                per_series=metrics,
                all_series_passed=all(m["passed"] for m in metrics),
                identical_input_output_exact=first == digest,
            )
        self.calls.append(row)

    def stage(self, *args):
        raise ValueError("The native monolithic path must not use the sequential adapter")

    def finish(self):
        from scripts.diagnostics.chronos2_frozen_validation import sha256

        matched = [row for row in self.calls if "per_series" in row]
        for name, row in self.files.items():
            path = self.root / name
            row.update(bytes=path.stat().st_size, sha256=sha256(path))
        report = {
            "scope": "Original full-model predictions, unchanged CPU oracle/criteria; per-series supplement",
            "reference_calls": self.reference_calls,
            "unique_reference_inputs": len(self.references),
            "calls": self.calls,
            "matched_calls": len(matched),
            "all_series_passed": bool(matched) and all(row["all_series_passed"] for row in matched),
            "identical_input_outputs_exact": bool(matched)
            and all(row["identical_input_output_exact"] for row in matched),
            "checked_CPU_oracle_models": self.checked_model_count,
            "files": self.files,
            "criteria": {"nrmse_max": 0.05, "pcc_min": 0.99, "zero_variance_reference": "exact equality"},
        }
        (self.root / "index.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
