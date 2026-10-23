import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoProcessor
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import BaseModelOutput

from optimum.rbln import RBLNSiglip2Model, RBLNSiglip2ModelConfig
from optimum.rbln.transformers.models.siglip2.modeling_siglip2 import patch_position_ids

from .siglip2_embedding_precision import EmbeddedInput, compare, flatten, tensor_hash


class AttentionBranch(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.norm = layer.layer_norm1
        self.attention = layer.self_attn

    def forward(self, hidden_states, pixel_attention_mask):
        hidden_states = self.norm(hidden_states)
        mask = create_bidirectional_mask(
            config=self.attention.config, inputs_embeds=hidden_states, attention_mask=pixel_attention_mask
        )
        return self.attention(hidden_states, attention_mask=mask)[0]


class MLPBranch(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.norm = layer.layer_norm2
        self.mlp = layer.mlp

    def forward(self, hidden_states):
        return self.mlp(self.norm(hidden_states))


class MLPProbe(MLPBranch):
    def forward(self, hidden_states):
        normalized = self.norm(hidden_states)
        projected = self.mlp.fc1(normalized)
        activated = self.mlp.activation_fn(projected)
        return self.mlp.fc2(activated), normalized, projected, activated


class AffineNormalization(nn.Module):
    def __init__(self, normalization):
        super().__init__()
        self.weight = normalization.weight
        self.bias = normalization.bias

    def forward(self, hidden_states):
        return hidden_states * self.weight + self.bias


class EmbeddedEncoder(nn.Module):
    def forward(self, inputs_embeds, attention_mask=None, **kwargs):
        return BaseModelOutput(last_hidden_state=inputs_embeds)


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
    embedding = wrapper.model.vision_model.embeddings
    layers = list(wrapper.model.vision_model.encoder.layers)
    branches = [(AttentionBranch(layer).eval(), MLPBranch(layer).eval()) for layer in layers]
    probes = {index: MLPProbe(layers[index]).eval() for index in [0, 6, 11]}
    fixture = hf_hub_download(
        "huggingface/documentation-images",
        "coco_sample.png",
        repo_type="dataset",
        revision="a9c3851af92cce883828811163f904b61a2550dc",
    )
    cat = Image.open(fixture).convert("RGB")
    images = [cat, cat.resize((256, 256)), cat.resize((128, 512)), cat.resize((512, 128))]
    texts = ["a photo of cats", "a photo of dogs", "a photo of a train", "a photo of parrots"]
    hashes = json.loads(Path(__file__).with_name("siglip2_original_input_hashes.json").read_text())
    inputs, references, cpu_baselines, cpu_hidden, cpu_stages = [], [], [], [], []
    with torch.no_grad():
        for index, (image, text) in enumerate(zip(images, texts, strict=True)):
            inp = processor(
                images=[image], text=[text], padding="max_length", max_length=64, truncation=True, return_tensors="pt"
            )
            if index == 0:
                for key, expected_hash in hashes.items():
                    assert tensor_hash(inp[key]) == expected_hash["sha256"], key
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
            baseline = tuple(value.clone() for value in wrapper(*values))
            for a, e in zip(baseline, expected, strict=True):
                torch.testing.assert_close(a, e)
            hidden = embedding(values[1], values[4])
            stages = []
            for layer_index, (layer, (attention, mlp)) in enumerate(zip(layers, branches, strict=True)):
                mask = create_bidirectional_mask(
                    config=layer.self_attn.config, inputs_embeds=hidden, attention_mask=values[3]
                )
                full = layer(hidden, mask)
                hidden = hidden + attention(hidden, values[3])
                mlp_output = mlp(hidden)
                if layer_index in probes:
                    torch.testing.assert_close(probes[layer_index](hidden)[0], mlp_output, rtol=0, atol=0)
                hidden = hidden + mlp_output
                torch.testing.assert_close(hidden, full, rtol=0, atol=0)
                stages.append(hidden.clone())
            cpu_stages.append(stages)
            inputs.append(values)
            references.append(expected)
            cpu_baselines.append(baseline)
            cpu_hidden.append(hidden)
        original_encoder = wrapper.model.vision_model.encoder
        wrapper.model.vision_model.embeddings = EmbeddedInput()
        wrapper.model.vision_model.encoder = EmbeddedEncoder()
        for values, hidden, baseline in zip(inputs, cpu_hidden, cpu_baselines, strict=True):
            for a, e in zip(wrapper(values[0], hidden, *values[2:]), baseline, strict=True):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
        normalization = wrapper.model.vision_model.post_layernorm
        wrapper.model.vision_model.post_layernorm = AffineNormalization(normalization)
        normalization_deltas = []
        for values, hidden, baseline in zip(inputs, cpu_hidden, cpu_baselines, strict=True):
            normalized = torch.nn.functional.layer_norm(hidden, normalization.normalized_shape, eps=normalization.eps)
            candidate = wrapper(values[0], normalized, *values[2:])
            deltas = []
            for a, e in zip(candidate, baseline, strict=True):
                torch.testing.assert_close(a, e)
                deltas.append(float((a - e).abs().max()))
            normalization_deltas.append(deltas)
        wrapper.model.vision_model.post_layernorm = normalization
        print("SIGLIP2_RESIDUAL_CPU_BITWISE four native geometries and all encoder residuals passed", flush=True)
        print("SIGLIP2_NORMALIZATION_CPU " + json.dumps(normalization_deltas), flush=True)
        if args.cpu_only:
            return

        import rebel

        assert importlib.metadata.version("rebel-compiler") == "0.12.1.dev328+g970308c6"
        out = Path("debug-siglip2-residual-precision")
        out.mkdir(exist_ok=True)
        report = {"compiler": importlib.metadata.version("rebel-compiler"), "compile": {}, "cases": []}
        paths = {}

        def compile_model(name, model, input_info):
            start = time.perf_counter()
            compiled = rebel.compile_from_torch(model, input_info=input_info, npu="RBLN-CR13", num_devices=1)
            path = out / f"{name}.rbln"
            compiled.save(path)
            executor = compiled.select_executor()
            names = [executor.get_input_profile(index).name for index in range(executor.num_inputs)]
            report["compile"][name] = {
                "seconds": time.perf_counter() - start,
                "bytes": compiled.get_total_device_alloc(),
                "inputs": names,
            }
            paths[name] = path
            print("SIGLIP2_RESIDUAL_COMPILE " + json.dumps({"name": name, **report["compile"][name]}), flush=True)
            (out / "results.json").write_text(json.dumps(report, indent=2))

        wrapper.model.vision_model.embeddings = embedding
        wrapper.model.vision_model.encoder = original_encoder
        compile_model("baseline", wrapper, config.compile_cfgs[0].input_info)
        compile_model(
            "embedding", embedding, [("pixel_values", [1, 256, 768], "float32"), ("position_ids", [1, 256], "int64")]
        )
        for index, (attention, mlp) in enumerate(branches):
            compile_model(
                f"attention-{index}",
                attention,
                [("hidden_states", [1, 256, 768], "float32"), ("pixel_attention_mask", [1, 256], "int64")],
            )
            compile_model(f"mlp-{index}", mlp, [("hidden_states", [1, 256, 768], "float32")])
        for index, probe in probes.items():
            compile_model(f"mlp-probe-{index}", probe, [("hidden_states", [1, 256, 768], "float32")])
        wrapper.model.vision_model.embeddings = EmbeddedInput()
        wrapper.model.vision_model.encoder = EmbeddedEncoder()
        compile_model("head", wrapper, config.compile_cfgs[0].input_info)
        assert report["compile"]["head"]["inputs"] == [
            item[0] for item in config.compile_cfgs[0].input_info if item[0] != "patch_position_ids"
        ]

        wrapper.model.vision_model.post_layernorm = AffineNormalization(normalization)
        compile_model("affine-head", wrapper, config.compile_cfgs[0].input_info)
        assert report["compile"]["affine-head"]["inputs"] == report["compile"]["head"]["inputs"]
        report["normalization_cpu_max_deltas"] = normalization_deltas
        report["timing_includes_cpu_diagnostics"] = True
        for variant in [
            "baseline",
            "fp32-residual",
            "fp32-residual-normalization",
            "diagnostic-cpu-attention",
            "diagnostic-cpu-mlp",
        ]:
            selected = (
                ["baseline"]
                if variant == "baseline"
                else [
                    key
                    for key in paths
                    if key not in {"baseline", "head" if variant == "fp32-residual-normalization" else "affine-head"}
                ]
            )
            compiled_models = {key: rebel.RBLNCompiledModel(paths[key]) for key in selected}
            runtimes = {
                key: model.create_runtime(tensor_type="pt", device=0) for key, model in compiled_models.items()
            }

            def execute(values, expected_stages=None, case_index=None, variant=variant, runtimes=runtimes):
                if variant == "baseline":
                    return tuple(value.clone() for value in runtimes["baseline"](*values)), None, []
                hidden = runtimes["embedding"](values[1], values[4])
                boundaries = []

                def nrmse(actual, reference):
                    return float((actual - reference).square().mean().sqrt() / reference.square().mean().sqrt())

                for index, (attention, mlp) in enumerate(branches):
                    cpu_attention = (
                        attention(hidden, values[3])
                        if expected_stages is not None or variant == "diagnostic-cpu-attention"
                        else None
                    )
                    attention_output = (
                        cpu_attention
                        if variant == "diagnostic-cpu-attention"
                        else runtimes[f"attention-{index}"](hidden, values[3]).float()
                    )
                    row = {"layer": index}
                    if expected_stages is not None:
                        row["attention_same_input_nrmse"] = nrmse(attention_output, cpu_attention)
                    hidden = hidden.float() + attention_output
                    cpu_mlp = mlp(hidden) if expected_stages is not None or variant == "diagnostic-cpu-mlp" else None
                    mlp_output = (
                        cpu_mlp if variant == "diagnostic-cpu-mlp" else runtimes[f"mlp-{index}"](hidden).float()
                    )
                    if expected_stages is not None:
                        row["mlp_same_input_nrmse"] = nrmse(mlp_output, cpu_mlp)
                        if variant == "fp32-residual" and index in probes:
                            probed = tuple(value.clone() for value in runtimes[f"mlp-probe-{index}"](hidden))
                            row["probe_matches_original"] = torch.equal(probed[0], mlp_output)
                            probe = probes[index]
                            component_expected = (
                                probe.mlp.fc2(probed[3]),
                                probe.norm(hidden),
                                probe.mlp.fc1(probed[1]),
                                probe.mlp.activation_fn(probed[2]),
                            )
                            row["mlp_components_same_input_nrmse"] = {
                                name: nrmse(a, e)
                                for name, a, e in zip(
                                    ["fc2", "normalization", "fc1", "activation"],
                                    probed,
                                    component_expected,
                                    strict=True,
                                )
                            }
                            torch.save(
                                {
                                    "input": hidden.clone(),
                                    "npu": probed,
                                    "component_expected": component_expected,
                                    "original_mlp": mlp_output.clone(),
                                },
                                out / f"mlp-probe-case{case_index}-layer{index}.pt",
                            )
                    hidden = hidden.float() + mlp_output
                    if expected_stages is not None:
                        row["hidden_total_nrmse"] = nrmse(hidden, expected_stages[index])
                        boundaries.append(row)
                hidden_before_head = hidden.clone()
                head = "head"
                if variant == "fp32-residual-normalization":
                    hidden = torch.nn.functional.layer_norm(
                        hidden, normalization.normalized_shape, eps=normalization.eps
                    )
                    head = "affine-head"
                actual = tuple(
                    value.clone() for value in runtimes[head](values[0], hidden, values[2], values[3], values[5])
                )
                return actual, hidden_before_head, boundaries

            first = None
            for index, (values, expected) in enumerate(zip(inputs, references, strict=True)):
                start = time.perf_counter()
                actual, hidden_before_head, boundaries = execute(values, cpu_stages[index], index)
                seconds = time.perf_counter() - start
                metrics = compare(actual, expected, execute(values)[0])
                report["cases"].append(
                    {
                        "variant": variant,
                        "case": index,
                        "seconds": seconds,
                        "metrics": metrics,
                        "boundaries": boundaries,
                    }
                )
                if hidden_before_head is not None:
                    cpu_postnorm = native.vision_model.post_layernorm(hidden_before_head)
                    report["cases"][-1]["cpu_postnorm_nrmse"] = float(
                        (cpu_postnorm - expected[6]).square().mean().sqrt() / expected[6].square().mean().sqrt()
                    )
                print("SIGLIP2_RESIDUAL_PARITY " + json.dumps(report["cases"][-1]), flush=True)
                torch.save(
                    {
                        "inputs": values,
                        "expected": expected,
                        "actual": actual,
                        "hidden_before_head": hidden_before_head,
                    },
                    out / f"{variant}-case{index}.pt",
                )
                (out / "results.json").write_text(json.dumps(report, indent=2))
                if first is None:
                    first = actual
            for a, e in zip(execute(inputs[0])[0], first, strict=True):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
            runtimes.clear()
            compiled_models.clear()
            gc.collect()
        report["diagnostic_only_variants"] = ["diagnostic-cpu-attention", "diagnostic-cpu-mlp"]
        (out / "results.json").write_text(json.dumps(report, indent=2))
        failures = [
            row
            for case in report["cases"]
            if case["variant"] == "fp32-residual-normalization"
            for row in case["metrics"]
            if row["failed_criteria"]
        ]
        assert not failures, failures


if __name__ == "__main__":
    main()
