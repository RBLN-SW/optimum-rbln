import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import rebel
import torch
from chronos import Chronos2Model

from optimum.rbln.chronos.chronos2_architecture import Chronos2ModelWrapper


class Features(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.core = Chronos2ModelWrapper(model)
        self.values = {}
        self.names = []
        for index, block in enumerate(model.encoder.block):
            self.add_hook(f"block_{index}", block, block_output=True)
        self.add_hook("pre_final_norm", model.encoder.final_layer_norm, before=True)
        self.add_hook("final_norm", model.encoder.final_layer_norm)
        self.add_hook("head_input", model.output_patch_embedding, before=True)
        for name in ["hidden_layer", "output_layer", "residual_layer"]:
            self.add_hook(name, getattr(model.output_patch_embedding, name))
        self.add_hook("head_sum", model.output_patch_embedding)

    def add_hook(self, name, module, before=False, block_output=False):
        self.names.append(name)

        def capture(module, inputs, output=None):
            value = inputs[0] if before else output.hidden_states if block_output else output
            self.values[name] = value[:, -65:]

        if before:
            module.register_forward_pre_hook(capture)
        else:
            module.register_forward_hook(capture)

    def forward(self, patched_context, context_mask, patched_future, group_ids):
        self.values = {}
        output = self.core(patched_context, context_mask, patched_future, group_ids)
        return (output, *(self.values[name] for name in self.names))


def metrics(actual, expected):
    actual, expected = actual.float(), expected.float()
    return {
        "max_delta": float((actual - expected).abs().max()),
        "nrmse": float((actual - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)),
        "actual_range": [float(actual.min()), float(actual.max())],
    }


def unpatch(value):
    return value.reshape(64, 64, 21, 16).permute(0, 2, 1, 3).reshape(64, 21, 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = Path("/mnt/shared_data/users/jaehunryu/nas_data/model-enable-20261024/chronos-allmissing")
    out = Path("debug-chronos2-allmissing-precision")
    out.mkdir(exist_ok=True)
    saved = torch.load(root / "failure-b64-ci766.pt", weights_only=False)
    inputs = saved["records"][-1]["inputs"]
    hashes = [hashlib.sha256(value.numpy().tobytes()).hexdigest() for value in inputs]
    compiler = importlib.metadata.version("rebel-compiler")
    report = {"compiler": compiler, "input_hashes": hashes, "cpu_only": args.cpu_only}
    if not args.cpu_only:
        assert compiler == "0.12.1.dev327+g00334c8e", compiler
        original = rebel.RBLNCompiledModel(root / "original-b64-ci766.rbln")
        runtime = original.create_runtime(device=0, tensor_type="pt")
        first = runtime(*inputs).clone()
        repeated = runtime(*inputs).clone()
        report["original_repeat_equal"] = torch.equal(first, repeated)
        report["original_matches_ci"] = torch.equal(first, saved["records"][-1]["output"])
        report["original_vs_ci"] = metrics(first, saved["records"][-1]["output"])
        del runtime, original
        gc.collect()

    model = (
        Chronos2Model.from_pretrained("amazon/chronos-2", revision="29ec3766d36d6f73f0696f85560a422f50e8498c")
        .float()
        .eval()
    )
    traced = Features(model).eval()
    with torch.no_grad():
        expected = tuple(value.clone() for value in traced(*inputs))
        _, _, loc_scale = model._prepare_patched_context(saved["inputs"]["context"], None)

        def inverse(value):
            return model.instance_norm.inverse(value.flatten(1), loc_scale).reshape(64, 21, 1024)

        report["cpu_vs_ci_forecast"] = metrics(inverse(expected[0]), saved["expected"])
        if args.cpu_only:
            actual = expected
        else:
            names = ["patched_context", "context_mask", "patched_future", "group_ids"]
            info = [
                (name, list(value.shape), str(value.dtype).removeprefix("torch."))
                for name, value in zip(names, inputs, strict=True)
            ]
            compiled = rebel.compile_from_torch(traced, input_info=info, npu="RBLN-CA25", num_devices=1)
            compiled.save(str(out / "instrumented.rbln"))
            runtime = compiled.create_runtime(device=0, tensor_type="pt")
            actual = tuple(value.clone() for value in runtime(*inputs))
            repeated = runtime(*inputs)
            report["instrumented_repeat_equal"] = all(torch.equal(a, b) for a, b in zip(actual, repeated, strict=True))
            report["instrumented_matches_ci"] = torch.equal(actual[0], saved["records"][-1]["output"])
            torch.save(dict(zip(["prediction", *traced.names], actual, strict=True)), out / "npu-features.pt")
        report["outputs"] = {
            name: metrics(a, e) for name, a, e in zip(["prediction", *traced.names], actual, expected, strict=True)
        }
        values = dict(zip(traced.names, actual[1:], strict=True))
        hybrid_head = unpatch(model.output_patch_embedding(values["head_input"]))
        hybrid_norm_head = unpatch(
            model.output_patch_embedding(model.encoder.final_layer_norm(values["pre_final_norm"])[:, -64:])
        )
        report["hybrid_head_forecast"] = metrics(inverse(hybrid_head), saved["expected"])
        report["hybrid_norm_head_forecast"] = metrics(inverse(hybrid_norm_head), saved["expected"])
        report["instrumented_forecast"] = metrics(inverse(actual[0]), saved["expected"])
        torch.save(dict(zip(["prediction", *traced.names], expected, strict=True)), out / "cpu-features.pt")
    report["inputs_unchanged"] = hashes == [hashlib.sha256(value.numpy().tobytes()).hexdigest() for value in inputs]
    (out / "results.json").write_text(json.dumps(report, indent=2))
    print("CHRONOS2_PRECISION " + json.dumps(report), flush=True)
    assert report["inputs_unchanged"]
    if not args.cpu_only:
        assert report["original_repeat_equal"]
        assert report["original_matches_ci"]
        assert report["instrumented_repeat_equal"]
        assert report["instrumented_matches_ci"], "Instrumentation changed the original failing output"


if __name__ == "__main__":
    main()
