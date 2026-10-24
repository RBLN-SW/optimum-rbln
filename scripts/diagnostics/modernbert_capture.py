"""Capture original native B1 outputs before unchanged ModernBERT parity gates."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import torch


SDK = "0.12.1.dev163+g0c6d4a5e.prod"
SDK249 = "0.12.1.dev249+gdef44cb7.prod"


def key(inputs):
    digest = hashlib.sha256()
    for name in ("input_ids", "attention_mask"):
        digest.update(inputs[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def hidden(output):
    return output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]


class Capture:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.references = {}
        self.rows = []
        self.outputs = {}
        self.active = False
        self.reference_model = None
        self.collected = 0
        self.failed = []
        self.skipped = []
        self.supplement_errors = []

    def event(self, **row):
        with (self.directory / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def reference(self, inputs, output):
        identity = key(inputs)
        if identity not in self.references:
            self.references[identity] = (
                {name: value.detach().cpu().clone() for name, value in inputs.items()},
                hidden(output).detach().cpu().clone(),
            )

    def actual(self, inputs, output):
        identity = key(inputs)
        if identity not in self.references or identity in self.outputs:
            return
        inputs, expected = self.references[identity]
        actual = hidden(output).detach().cpu().clone()
        offset = len(self.rows)
        raw = self.directory / f"offset-{offset}.pt"
        # Persist before computing metrics or allowing the original assertion.
        torch.save({"inputs": inputs, "reference": expected, "actual": actual}, raw)
        self.outputs[identity] = actual
        mask = inputs["attention_mask"][0].bool()
        target = expected[0, mask].float().flatten()
        value = actual[0, mask].float().flatten()

        def pool(tensor):
            return torch.nn.functional.normalize(
                (tensor * inputs["attention_mask"][..., None]).sum(1) / inputs["attention_mask"].sum(1, keepdim=True),
                dim=1,
            )

        pcc = torch.nn.functional.cosine_similarity(value - value.mean(), target - target.mean(), dim=0).item()
        nrmse = ((value - target).square().mean().sqrt() / target.square().mean().sqrt()).item()
        cosine = torch.nn.functional.cosine_similarity(pool(actual)[0], pool(expected)[0], dim=0).item()
        row = {
            "offset": offset,
            "input_sha256": identity,
            "raw": raw.name,
            "valid_tokens": int(mask.sum()),
            "pcc": pcc,
            "nrmse": nrmse,
            "cosine": cosine,
            "finite": bool(torch.isfinite(actual).all()),
        }
        row["threshold_passed"] = row["finite"] and pcc >= 0.99 and nrmse <= 0.05 and cosine >= 0.99
        self.rows.append(row)
        (self.directory / "metrics.json").write_text(json.dumps(self.rows, indent=2) + "\n")

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.failed.append(report.nodeid)
        if report.skipped:
            self.skipped.append(report.nodeid)

    def supplement_after_failure(self, case):
        from transformers import AutoTokenizer

        if self.reference_model is None:
            raise AssertionError("No matched CPU reference captured")
        tokenizer = AutoTokenizer.from_pretrained(case.HF_MODEL_ID, **case.HF_CONFIG_KWARGS)
        with torch.inference_mode():
            for text in case.TEXTS:
                inputs = tokenizer([text], max_length=8192, padding="max_length", truncation=True, return_tensors="pt")
                inputs.pop("token_type_ids", None)
                if key(inputs) not in self.outputs:
                    self.reference_model(**inputs)
                    case.model(**inputs, return_dict=True)
            pairs = list(self.references.items())
            first_key, (first_inputs, _) = pairs[0]
            last_key, (last_inputs, _) = pairs[-1]
            first = self.outputs[first_key]
            last = self.outputs[last_key]
            assert torch.equal(last, hidden(case.model(**last_inputs, return_dict=True)))
            assert torch.equal(last, case.model(**last_inputs, return_dict=False)[0])
            assert torch.equal(first, hidden(case.model(**first_inputs, return_dict=True)))
            del type(case).model
            gc.collect()
            type(case).model = case.RBLN_AUTO_CLASS.from_pretrained(case.get_rbln_local_dir(), rbln_device=case.DEVICE)
            assert torch.equal(first, hidden(case.model(**first_inputs, return_dict=True)))
            assert torch.equal(last, hidden(case.model(**last_inputs, return_dict=True)))
        self.event(phase="supplement", four_offsets=True, repeat_tuple_return_reload=True)


def run(directory, arm):
    import pytest
    from transformers import AutoModel

    from optimum.rbln import RBLNModernBertModel
    from tests import test_transformers

    capture = Capture(directory)
    case = test_transformers.TestModernBertModelNativeMaxSeqLen
    original_test = case.test_embedding_parity
    original_load = AutoModel.from_pretrained
    original_forward = RBLNModernBertModel.forward
    original_wrap = RBLNModernBertModel._wrap_model_if_needed

    def load_reference(*args, **kwargs):
        result = original_load(*args, **kwargs)
        if capture.active and kwargs.get("output_loading_info"):
            model = result[0]
            forward = model.forward

            def reference_forward(*args, **kwargs):
                result = forward(*args, **kwargs)
                capture.reference(kwargs, result)
                return result

            model.forward = reference_forward
            capture.reference_model = model
        return result

    def actual_forward(self, *args, **kwargs):
        start = time.perf_counter()
        error = None
        try:
            output = original_forward(self, *args, **kwargs)
            if capture.active:
                inputs = {
                    name: kwargs.get(name, args[index] if len(args) > index else None)
                    for index, name in enumerate(("input_ids", "attention_mask"))
                }
                capture.actual(inputs, output)
            return output
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            capture.event(phase="forward", seconds=time.perf_counter() - start, error=error)

    def numerical(self):
        capture.active = True
        try:
            original_test(self)
        except AssertionError:
            original_failure = traceback.format_exc()
            (directory / "original-assertion.txt").write_text(original_failure)
            try:
                capture.supplement_after_failure(self)
            except BaseException:
                capture.supplement_errors.append(traceback.format_exc())
            raise
        finally:
            capture.active = False

    def wrapper(cls, model, config):
        if arm == "cached-rope":
            from .modernbert_rotary import install_fixed_rotary

            install_fixed_rotary(model, 8192)
        return original_wrap(model, config)

    with (
        patch.object(AutoModel, "from_pretrained", side_effect=load_reference),
        patch.object(RBLNModernBertModel, "forward", actual_forward),
        patch.object(RBLNModernBertModel, "_wrap_model_if_needed", classmethod(wrapper)),
        patch.object(case, "test_embedding_parity", numerical),
    ):
        code = int(
            pytest.main(
                [
                    "tests/test_transformers.py::TestModernBertModelNativeMaxSeqLen",
                    "-vv",
                    "-s",
                    "-o",
                    "addopts=",
                    f"--junitxml={directory / 'junit.xml'}",
                ],
                plugins=[capture],
            )
        )
    if capture.collected != 9 or capture.skipped or len(capture.rows) != 4 or capture.supplement_errors:
        code = 1
    return code, capture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["original", "cached-rope"], default="original")
    parser.add_argument("--output", type=Path, default=Path("debug-modernbert-capture"))
    parser.add_argument("--expected-sdk", choices=[SDK, SDK249], default=SDK)
    parser.add_argument("--execute", action="store_true", help="Scheduler-only real NPU execution")
    args = parser.parse_args(argv)
    directory = args.output.resolve() / args.arm
    directory.mkdir(parents=True, exist_ok=True)
    result = {"arm": args.arm, "accepted": False, "phase": "preflight"}
    code = 1
    try:
        if not args.execute:
            raise ValueError("--execute is required; run only in the scheduled CR13 lane")
        if os.environ.get("SAVE_ARTIFACTS_PATH") or os.environ.get("REUSE_ARTIFACTS_PATH"):
            raise ValueError("Artifact compatibility mode cannot run this diagnostic")
        result["sdk"] = importlib.metadata.version("rebel-compiler")
        if result["sdk"] != args.expected_sdk:
            raise ValueError(f"Expected diagnostic SDK {args.expected_sdk}")
        os.environ["RBLN_TEST_NPU"] = "RBLN-CR13"
        os.environ["OPTIMUM_RBLN_TEST_LEVEL"] = "full"
        result["phase"] = "tests"
        code, capture = run(directory, args.arm)
        result.update(
            phase="finished",
            exit_code=code,
            collected=capture.collected,
            failed=capture.failed,
            skipped=capture.skipped,
            metrics=capture.rows,
            supplement_errors=capture.supplement_errors,
            original_tests_passed=code == 0,
        )
    except BaseException:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        result["exit_code"] = code
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if not (directory / "junit.xml").exists():
            suite = ET.Element(
                "testsuite", name="modernbert_capture_preflight", tests="1", failures=str(int(code != 0))
            )
            case = ET.SubElement(suite, "testcase", name=result["phase"])
            if code:
                ET.SubElement(case, "failure").text = result.get("error", "Diagnostic failed")
            ET.ElementTree(suite).write(directory / "junit.xml", encoding="unicode")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
