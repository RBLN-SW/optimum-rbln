import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoProcessor

from optimum.rbln import RBLNSiglip2Model, RBLNSiglip2ModelConfig
from optimum.rbln.transformers.models.siglip2.modeling_siglip2 import patch_position_ids


class EmbeddingParts(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, pixel_values, position_ids):
        return self.embeddings.patch_embedding(pixel_values), self.embeddings.position_embedding(position_ids)


class EmbeddedInput(nn.Module):
    def forward(self, pixel_values, spatial_shapes):
        return pixel_values


def flatten(output):
    values = output.to_tuple()
    return tuple(value.clone() for value in (*values[:4], *values[4], *values[5]))


def tensor_hash(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def compare(actual, expected, repeated):
    rows = []
    for index, (a, e, r) in enumerate(zip(actual, expected, repeated, strict=True)):
        nrmse = float((a - e).square().mean().sqrt() / e.square().mean().sqrt().clamp_min(1e-8))
        row = {"output": index, "nrmse": nrmse, "repeat": torch.equal(a, r)}
        failures = []
        if not a.isfinite().all():
            failures.append("finite")
        if nrmse > 0.05:
            failures.append("nrmse")
        if not row["repeat"]:
            failures.append("repeat")
        if a.numel() > 1:
            row["pcc"] = float(torch.corrcoef(torch.stack([a.flatten(), e.flatten()]))[0, 1])
            if not row["pcc"] >= 0.99:
                failures.append("pcc")
        if index in (2, 3):
            row["minimum_cosine"] = float(torch.nn.functional.cosine_similarity(a, e).min())
            if row["minimum_cosine"] < 0.99:
                failures.append("cosine")
        row["failed_criteria"] = failures
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = "google/siglip2-base-patch16-naflex"
    revision = "b53b807d3a2d5e2b3911292f2d69e5341cdc064c"
    native = AutoModel.from_pretrained(checkpoint, revision=revision).float().eval()
    processor = AutoProcessor.from_pretrained(checkpoint, revision=revision)
    config = RBLNSiglip2Model._update_rbln_config(None, native, native.config, RBLNSiglip2ModelConfig())
    wrapper = RBLNSiglip2Model._wrap_model_if_needed(native, config)
    parts = EmbeddingParts(wrapper.model.vision_model.embeddings).eval()
    original_embedding = wrapper.model.vision_model.embeddings
    fixture = hf_hub_download(
        "huggingface/documentation-images",
        "coco_sample.png",
        repo_type="dataset",
        revision="a9c3851af92cce883828811163f904b61a2550dc",
    )
    cat = Image.open(fixture).convert("RGB")
    images = [cat, cat.resize((256, 256)), cat.resize((128, 512)), cat.resize((512, 128))]
    texts = ["a photo of cats", "a photo of dogs", "a photo of a train", "a photo of parrots"]
    original_hashes = json.loads(Path(__file__).with_name("siglip2_original_input_hashes.json").read_text())
    inputs = []
    references = []
    cpu_deltas = []
    with torch.no_grad():
        for index, (image, text) in enumerate(zip(images, texts, strict=True)):
            inp = processor(
                images=[image], text=[text], padding="max_length", max_length=64, truncation=True, return_tensors="pt"
            )
            if index == 0:
                for key, expected in original_hashes.items():
                    assert list(inp[key].shape) == expected["shape"]
                    assert str(inp[key].dtype) == expected["dtype"]
                    assert tensor_hash(inp[key]) == expected["sha256"], key
            ids = patch_position_ids(inp.spatial_shapes, 256)
            values = (
                inp.input_ids,
                inp.pixel_values,
                inp.get("attention_mask", torch.ones_like(inp.input_ids)),
                inp.pixel_attention_mask.long(),
                ids,
                torch.arange(64).unsqueeze(0),
            )
            expected = flatten(native(**inp))
            baseline_cpu = tuple(value.clone() for value in wrapper(*values))
            deltas = []
            for a, e in zip(baseline_cpu, expected, strict=True):
                torch.testing.assert_close(a, e)
                deltas.append(float((a - e).abs().max()))
            cpu_deltas.append(deltas)
            patch, position = parts(inp.pixel_values, ids)
            wrapper.model.vision_model.embeddings = EmbeddedInput()
            candidate_values = (values[0], patch.float() + position.float(), *values[2:])
            for a, e in zip(wrapper(*candidate_values), baseline_cpu, strict=True):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
            wrapper.model.vision_model.embeddings = original_embedding
            inputs.append(values)
            references.append(expected)
    print(
        "SIGLIP2_SPLIT_CPU_BITWISE four grids preserve existing CPU wrapper bitwise; native references and original failed input hashes verified",
        flush=True,
    )
    if args.cpu_only:
        return

    import rebel

    assert importlib.metadata.version("rebel-compiler") == "0.12.1.dev328+g970308c6"
    out = Path("debug-siglip2-embedding-precision")
    out.mkdir(exist_ok=True)
    report = {
        "compiler": importlib.metadata.version("rebel-compiler"),
        "cases": [],
        "compile": {},
        "cpu_native_max_deltas": cpu_deltas,
    }
    artifacts = {}
    with torch.no_grad():
        for name, model, info in [
            ("baseline", wrapper, config.compile_cfgs[0].input_info),
            ("parts", parts, [("pixel_values", [1, 256, 768], "float32"), ("position_ids", [1, 256], "int64")]),
        ]:
            start = time.perf_counter()
            compiled = rebel.compile_from_torch(model, input_info=info, npu="RBLN-CR13", num_devices=1)
            path = out / f"{name}.rbln"
            compiled.save(path)
            report["compile"][name] = {
                "seconds": time.perf_counter() - start,
                "bytes": compiled.get_total_device_alloc(),
            }
            artifacts[name] = path
            del compiled
            gc.collect()
        wrapper.model.vision_model.embeddings = EmbeddedInput()
        start = time.perf_counter()
        compiled = rebel.compile_from_torch(
            wrapper, input_info=config.compile_cfgs[0].input_info, npu="RBLN-CR13", num_devices=1
        )
        artifacts["body"] = out / "body.rbln"
        compiled.save(artifacts["body"])
        report["compile"]["body"] = {
            "seconds": time.perf_counter() - start,
            "bytes": compiled.get_total_device_alloc(),
        }
        del compiled
        gc.collect()

        first = {}
        for name in ["baseline", "split"]:
            runtimes = {}
            compiled_models = {}
            for component in ["baseline"] if name == "baseline" else ["parts", "body"]:
                runtimes[component] = rebel.RBLNCompiledModel(artifacts[component]).create_runtime(
                    tensor_type="pt", device=0
                )

            def execute(values, name=name, runtimes=runtimes):
                if name == "baseline":
                    return tuple(value.clone() for value in runtimes["baseline"](*values))
                patch, position = runtimes["parts"](values[1], values[4])
                embedded = patch.float() + position.float()
                return tuple(value.clone() for value in runtimes["body"](values[0], embedded, *values[2:]))

            for index, (values, expected) in enumerate(zip(inputs, references, strict=True)):
                start = time.perf_counter()
                actual = execute(values)
                seconds = time.perf_counter() - start
                repeated = execute(values)
                metrics = compare(actual, expected, repeated)
                report["cases"].append({"variant": name, "case": index, "seconds": seconds, "metrics": metrics})
                print("SIGLIP2_EMBEDDING_PRECISION " + json.dumps(report["cases"][-1]), flush=True)
                torch.save({"inputs": values, "expected": expected, "actual": actual}, out / f"{name}-case{index}.pt")
                (out / "results.json").write_text(json.dumps(report, indent=2))
                if index == 0:
                    first[name] = actual
            for a, e in zip(execute(inputs[0]), first[name], strict=True):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
            runtimes.clear()
            compiled_models.clear()
            gc.collect()
        failures = [
            row
            for case in report["cases"]
            if case["variant"] == "split"
            for row in case["metrics"]
            if row["failed_criteria"]
        ]
        assert not failures, failures


if __name__ == "__main__":
    main()
