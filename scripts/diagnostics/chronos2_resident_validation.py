"""Validate explicit residency modes of the existing six-stage full Chronos candidate."""

import argparse
import contextlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from scripts.diagnostics import chronos2_frozen_validation as original
from scripts.diagnostics.chronos2_candidate_validation import validate_candidate


class ResidentRuntimeChain:
    """Keep learned stages on device and copy only unmodified graph boundaries."""

    def __init__(self, runtimes, event):
        self.runtimes = runtimes
        self.event = event

    def __call__(self, *inputs):
        hidden, time_mask, group_mask = (x.clone() for x in self.runtimes[0](*inputs))
        for runtime in self.runtimes[1:-1]:
            hidden = runtime(hidden, time_mask, group_mask)
            if isinstance(hidden, (list, tuple)):
                if len(hidden) != 1:
                    raise ValueError("Encoder stage must return one full hidden tensor")
                hidden = hidden[0]
            hidden = hidden.clone()
        output = self.runtimes[-1](hidden)
        self.event({"phase": "resident_request_completed", "learned_stages": len(self.runtimes)})
        return output


def resident_adapter(manifest, event, observer=None):
    """Build every runtime from this invocation's compiled models, including reloads."""
    if observer is not None:
        raise ValueError("Prediction audit must not request a sequential stage observer")

    def create(cls, compiled_models, rbln_config):
        names = [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs]
        if names != list(original.NAMES) or len(compiled_models) != len(names):
            raise ValueError("All six original stages are required")
        devices = [rbln_config.device_map[name] for name in names]
        if any(device != devices[0] for device in devices):
            raise ValueError("All stages must use the same scheduler-owned device")
        runtimes = [
            model.create_runtime(
                tensor_type="pt",
                device=device,
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for model, device in zip(compiled_models, devices, strict=True)
        ]
        event({"phase": "resident_chain_created", "compiled_model_object_ids": [id(m) for m in compiled_models]})
        return [ResidentRuntimeChain(runtimes, event)]

    return create


@contextlib.contextmanager
def residency_context(audit, mode):
    """Sequential mode retains the original unload-before-next-stage adapter."""
    if mode not in ("resident", "sequential"):
        raise ValueError("Unknown residency mode")
    with contextlib.ExitStack() as stack:
        if mode == "resident":
            stack.enter_context(patch.object(audit, "stage", None))
            stack.enter_context(patch.object(original, "runtime_adapter", resident_adapter))
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--residency", choices=["resident", "sequential"], default="resident")
    args = parser.parse_args()
    args.artifacts = args.artifacts.resolve()
    manifest = validate_candidate(args.artifacts, args.manifest_sha256)
    if not os.environ.get("BUILDKITE_JOB_ID") or os.environ.get("RBLN_TEST_NPU") != "RBLN-CA25":
        raise ValueError("Execution requires the scheduler-owned CA25 job")
    if any(
        name in os.environ
        for name in ["RBLN_DUMMY_DEVICE", "TVM_LIBRARY_PATH", "SAVE_ARTIFACTS_PATH", "REUSE_ARTIFACTS_PATH"]
    ):
        raise ValueError("Dummy, library overrides and artifact compatibility modes are forbidden")
    runtime = original.verify_runtime_library("4e1bc153f001ae5a8694478052c7df4b05d8b5dbc09c6a27c06153e6decc7a9e")
    import rebel

    from scripts.diagnostics.chronos2_prediction_audit import PredictionAudit

    validate_candidate(args.artifacts, args.manifest_sha256, rebel.RBLNCompiledModel.inspect)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    weights = json.loads((args.artifacts / "compile-encoder_0.json").read_text())["checkpoint_weights"]
    audit = PredictionAudit(args.out / "predictions", weights)

    def event(value):
        with (args.out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print("CHRONOS_RESIDENT " + json.dumps(value), flush=True)

    event(
        {
            "phase": "verified_before_runtime",
            "runtime": runtime,
            "full_model": True,
            "product_adopted": False,
            "residency": args.residency,
        }
    )
    os.environ["OPTIMUM_RBLN_TEST_LEVEL"] = "full"
    os.chdir(args.out)
    # The full-model prediction audit is independent of stage-boundary observation.
    with residency_context(audit, args.residency):
        try:
            original.run_suite(args.artifacts, manifest, args.out, event, diagnostics=audit)
        finally:
            report = audit.finish()
            event(
                {
                    "phase": "prediction_audit",
                    "matched_calls": report["matched_calls"],
                    "reference_calls": len(report["reference_calls"]),
                    "unique_inputs": report["unique_reference_inputs"],
                    "all_series_passed": report["all_series_passed"],
                    "identical_input_outputs_exact": report["identical_input_outputs_exact"],
                }
            )
    if (
        report["matched_calls"] != 22
        or len(report["reference_calls"]) != 10
        or report["unique_reference_inputs"] != 5
        or not report["all_series_passed"]
        or not report["identical_input_outputs_exact"]
    ):
        raise RuntimeError("Resident whole-model/per-series/reload acceptance is incomplete")


if __name__ == "__main__":
    main()
