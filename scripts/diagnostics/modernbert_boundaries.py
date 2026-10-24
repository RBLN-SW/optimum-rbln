"""Full-model native B1 boundary diagnostic; never a model-acceptance substitute."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import torch


CHECKPOINT = "nomic-ai/modernbert-embed-base"
REVISION = "d556a88e332558790b210f7bdbe87da2fa94a8d8"
SDK = "0.12.1.dev163+g0c6d4a5e.prod"
BASELINE_SHA256 = "bed9721c662ca682645dee9469a2dbe17c3028559ec4a388c21a3e87f2b267e4"
BOUNDARIES = (14, 15, 16)  # Zero-based encoder layer indices.


class BoundaryWrapper(torch.nn.Module):
    def __init__(self, model, boundaries=BOUNDARIES):
        super().__init__()
        self.model = model
        self.boundaries = boundaries
        if any(index < 0 or index >= model.config.num_hidden_layers - 1 for index in boundaries):
            raise ValueError(
                "Requested boundary must precede the last encoder layer; HF replaces its hidden-state entry with final normalization"
            )

    def forward(self, input_ids, attention_mask):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True
        )
        return (output.last_hidden_state, *(output.hidden_states[index + 1] for index in self.boundaries))


def metrics(actual, expected):
    value, target = actual.double().flatten(), expected.double().flatten()
    return {
        "finite": bool(torch.isfinite(value).all()),
        "pcc": torch.nn.functional.cosine_similarity(value - value.mean(), target - target.mean(), dim=0).item(),
        "nrmse": ((value - target).square().mean().sqrt() / target.square().mean().sqrt()).item(),
        "max_abs": (value - target).abs().max().item(),
    }


def load_baseline(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise ValueError("Expected the exact SHA-verified CI806 offset-2 artifact")
    baseline = torch.load(path, weights_only=True)
    inputs = baseline["inputs"]
    if tuple(inputs["input_ids"].shape) != (1, 8192) or not bool(inputs["attention_mask"].all()):
        raise ValueError("Baseline must be original B1 with all 8192 valid tokens")
    return baseline


def compare_to_frozen(actual, baseline):
    # A changed final output invalidates causal use of the added boundaries.
    return {
        "bit_exact": torch.equal(actual, baseline),
        **metrics(actual, baseline),
    }


def verify_trained_state(model):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    checkpoint = Path(hf_hub_download(CHECKPOINT, "model.safetensors", revision=REVISION))
    state = model.state_dict()
    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        if set(state) != set(tensors.keys()):
            raise ValueError("Checkpoint state tensor names changed")
        for name, value in state.items():
            if value.dtype != torch.float32 or not torch.equal(value, tensors.get_tensor(name)):
                raise ValueError(f"Trained checkpoint tensor changed: {name}")
    return {
        "state_tensors": len(state),
        "parameters": sum(p.numel() for p in model.parameters()),
        "exact_checkpoint_tensors": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("debug-modernbert-boundaries"))
    parser.add_argument("--execute", action="store_true", help="Scheduled CR13 execution only")
    parser.add_argument("--compile-only", action="store_true", help="CPU compilation; creates no runtime")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "accepted": False,
        "phase": "preflight",
        "boundaries": ["final", *[f"after_layer_{i}" for i in BOUNDARIES]],
    }
    code = 1
    result.update(
        checkpoint=CHECKPOINT,
        revision=REVISION,
        baseline_sha256=BASELINE_SHA256,
        helper_sha256_at_start=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkout_head_at_start=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    try:
        if args.execute == args.compile_only:
            raise ValueError("Choose exactly one of --execute or --compile-only")
        if os.environ.get("SAVE_ARTIFACTS_PATH") or os.environ.get("REUSE_ARTIFACTS_PATH"):
            raise ValueError("Artifact compatibility mode cannot run this diagnostic")
        if args.execute and not os.environ.get("BUILDKITE_BUILD_ID"):
            raise ValueError("--execute requires the scheduler BUILDKITE_BUILD_ID")
        result["sdk"] = importlib.metadata.version("rebel-compiler")
        if result["sdk"] != SDK:
            raise ValueError("This diagnostic requires original SDK163.prod")
        baseline = load_baseline(args.baseline)
        from transformers import AutoModel

        from optimum.rbln import RBLNModernBertModel

        result["phase"] = "cpu_reference"
        reference, loading = AutoModel.from_pretrained(
            CHECKPOINT, revision=REVISION, attn_implementation="sdpa", output_loading_info=True
        )
        reference.eval()
        if (
            loading["missing_keys"]
            or loading["unexpected_keys"]
            or sum(p.numel() for p in reference.parameters()) != 149014272
        ):
            raise ValueError("Expected all149M trained checkpoint weights unchanged")
        result["reference_checkpoint_state"] = verify_trained_state(reference)
        recorded = {}
        handles = [
            reference.layers[index].register_forward_hook(
                lambda module, args, output, index=index: recorded.update({index: output.clone()})
            )
            for index in BOUNDARIES
        ]
        with torch.inference_mode():
            original_final = reference(**baseline["inputs"]).last_hidden_state
            for handle in handles:
                handle.remove()
            expected = [tensor.clone() for tensor in BoundaryWrapper(reference)(**baseline["inputs"])]
            torch.testing.assert_close(expected[0], original_final, rtol=0, atol=0)
            for index, tensor in zip(BOUNDARIES, expected[1:], strict=True):
                torch.testing.assert_close(tensor, recorded[index], rtol=0, atol=0)
        result["reference_checkpoint_state_after_forward"] = verify_trained_state(reference)
        result["cpu_full_model_equivalence"] = "Final and selected layers match normal full-model forward exactly"
        result["cpu_reference_against_frozen"] = metrics(expected[0], baseline["reference"])
        torch.save(
            {"inputs": baseline["inputs"], "reference": expected, "names": result["boundaries"]},
            args.output / "reference.pt",
        )
        del reference
        gc.collect()
        result["phase"] = "compile"

        def wrap(cls, native, config):
            result["compiled_checkpoint_state"] = verify_trained_state(native)
            return BoundaryWrapper(native).eval()

        with patch.object(RBLNModernBertModel, "_wrap_model_if_needed", classmethod(wrap)):
            model = RBLNModernBertModel.from_pretrained(
                CHECKPOINT,
                revision=REVISION,
                export=True,
                rbln_config={
                    "batch_size": 1,
                    "max_seq_len": 8192,
                    "npu": "RBLN-CR13",
                    "num_devices": 1,
                    "create_runtimes": args.execute,
                },
                model_save_dir=args.output / "model",
            )
        result["alloc_per_node"] = model.compiled_models[0].get_alloc_per_node()
        if args.compile_only:
            result["phase"] = "compiled_only"
            code = 0
        else:
            result["phase"] = "execute"
            inputs = baseline["inputs"]
            with torch.inference_mode():
                # The augmented graph deliberately exposes four tensors. Read its
                # ordinary runtime directly, without reinterpreting BaseModelOutput.
                actual = [tensor.clone() for tensor in model.model[0](inputs["input_ids"], inputs["attention_mask"])]
                torch.save(
                    {"inputs": inputs, "actual": actual, "names": result["boundaries"]}, args.output / "actual.pt"
                )
                result["frozen_final_comparison"] = compare_to_frozen(actual[0], baseline["actual"])
                result["metrics"] = [
                    {"name": name, **metrics(value, target)}
                    for name, value, target in zip(result["boundaries"], actual, expected, strict=True)
                ]
                repeated = model.model[0](inputs["input_ids"], inputs["attention_mask"])
                result["repeat_exact"] = all(torch.equal(a, b) for a, b in zip(actual, repeated, strict=True))
                del repeated
                model.save_pretrained(args.output / "reload")
                del model
                gc.collect()
                reloaded = RBLNModernBertModel.from_pretrained(args.output / "reload", rbln_device=0)
                output = reloaded.model[0](inputs["input_ids"], inputs["attention_mask"])
                result["reload_exact"] = all(torch.equal(a, b) for a, b in zip(actual, output, strict=True))
            final = result["metrics"][0]
            final["pooled_cosine"] = torch.nn.functional.cosine_similarity(
                actual[0].mean(1), expected[0].mean(1), dim=1
            ).item()
            result["original_final_threshold_passed"] = (
                final["finite"] and final["pcc"] >= 0.99 and final["nrmse"] <= 0.05 and final["pooled_cosine"] >= 0.99
            )
            result["instrumentation_preserved_original_final"] = result["frozen_final_comparison"]["bit_exact"]
            code = (
                0
                if result["instrumentation_preserved_original_final"]
                and result["original_final_threshold_passed"]
                and result["repeat_exact"]
                and result["reload_exact"]
                else 1
            )
            result["phase"] = "finished"
    except BaseException:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        result["exit_code"] = code
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        suite = ET.Element("testsuite", name="modernbert_boundary_diagnostic", tests="1", failures=str(int(code != 0)))
        case = ET.SubElement(suite, "testcase", name=result["phase"])
        if code:
            ET.SubElement(case, "failure").text = result.get(
                "error", "Original numerical gate or instrumentation identity/state check failed"
            )
        ET.ElementTree(suite).write(args.output / "junit.xml", encoding="unicode")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
