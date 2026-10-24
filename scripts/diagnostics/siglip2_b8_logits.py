"""Scheduler-only SigLIP2 B8 capture and frozen-input head discrimination.

No device is touched until main() receives --device. --analyze is CPU-only.
Run in the exact CI786 product/SDK environment. Never interprets hidden-output
failures as acceptance of the original product model.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import torch


MODEL = "google/siglip2-base-patch16-naflex"
REVISION = "b53b807d3a2d5e2b3911292f2d69e5341cdc064c"
FIXTURE_REVISION = "a9c3851af92cce883828811163f904b61a2550dc"
NAMES = [
    "image_logits",
    "text_logits",
    "text_embeds",
    "image_embeds",
    "text_hidden",
    "text_pool",
    "vision_hidden",
    "vision_pool",
]


def flatten(output):
    values = output.to_tuple()
    return tuple(v.detach().cpu().clone() for v in (*values[:4], *values[4], *values[5]))


def tensor_info(value):
    value = value.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
    }


def metrics(actual, expected):
    a, e = actual.float(), expected.float()
    result = {
        "finite": bool(a.isfinite().all()),
        "shape": list(a.shape),
        "nrmse": float((a - e).square().mean().sqrt() / e.square().mean().sqrt().clamp_min(1e-8)),
        "max_abs_error": float((a - e).abs().max()),
        "actual_range": [float(a.min()), float(a.max())],
        "expected_range": [float(e.min()), float(e.max())],
    }
    if a.numel() > 1:
        result["pcc"] = float(torch.corrcoef(torch.stack([a.flatten(), e.flatten()]))[0, 1])
    return result


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=True))


def analyze(path):
    d = torch.load(path, map_location="cpu", weights_only=True)
    a, e = d["actual"], d["expected"]
    scale, bias = d["logit_scale"].exp(), d["logit_bias"]
    npu_raw = a[2].float() @ a[3].float().T
    cpu_raw = e[2].float() @ e[3].float().T
    rebuilt = npu_raw * scale + bias
    reference_rebuilt = cpu_raw * scale + bias
    rows = {
        "scale": scale.tolist(),
        "bias": bias.tolist(),
        "reference_rebuild_vs_reference_text_logits": metrics(reference_rebuilt, e[1]),
        "npu_embed_cpu_rebuild_vs_reference_text_logits": metrics(rebuilt, e[1]),
        "actual_text_logits_vs_npu_embed_cpu_rebuild": metrics(a[1], rebuilt),
        "actual_image_logits_vs_npu_embed_cpu_rebuild_transpose": metrics(a[0], rebuilt.T),
        "actual_text_image_transpose_equal": bool(torch.equal(a[0], a[1].T)),
        "actual_text_logits": a[1].tolist(),
        "rebuilt_text_logits": rebuilt.tolist(),
        "reference_text_logits": e[1].tolist(),
    }
    # Fit affine calibration as a discriminator; never a product correction.
    design = torch.stack([npu_raw.flatten().double(), torch.ones(npu_raw.numel(), dtype=torch.float64)], 1)
    fit = torch.linalg.lstsq(design, a[1].flatten().double()).solution
    rows["actual_affine_fit_to_npu_raw"] = {
        "scale_bias": fit.tolist(),
        "residual": metrics((design @ fit).reshape_as(a[1]), a[1]),
    }
    # A 4+4 repeated fixture should retain the same row/column pattern.
    if a[1].shape == (8, 8):
        rows["repeat_4x4_max_abs"] = {
            name: float((value[:4, :4] - value[4:, 4:]).abs().max())
            for name, value in [("actual", a[1]), ("rebuilt", rebuilt), ("reference", e[1])]
        }
    dump_json(path.with_suffix(".analysis.json"), rows)
    return rows


class Head(torch.nn.Module):
    def __init__(self, scale, bias, mode):
        super().__init__()
        self.register_buffer("logit_scale", scale.clone())
        self.register_buffer("logit_bias", bias.clone())
        self.mode = mode

    def forward(self, text, image):
        raw = text @ image.t()
        calibrated = raw * self.logit_scale.exp() + self.logit_bias
        if self.mode == "raw":
            return raw
        if self.mode == "calibrated":
            return calibrated
        if self.mode == "pair":
            return calibrated.t(), calibrated
        return calibrated.t(), calibrated, raw, text, image


def head_controls(args, case):
    import rebel

    d = torch.load(case, map_location="cpu", weights_only=True)
    rows = []
    for batch in args.batches:
        for source in ("actual", "expected"):
            values = d[source]
            # Freeze actual checkpoint-derived inputs on disk. No random substitute.
            text = values[2].repeat((batch + 7) // 8, 1)[:batch].contiguous()
            image = values[3].repeat((batch + 7) // 8, 1)[:batch].contiguous()
            for mode in ("raw", "calibrated", "pair", "diagnostic"):
                label = f"head-b{batch}-{source}-{mode}"
                model = Head(d["logit_scale"], d["logit_bias"], mode).eval()
                expected = model(text, image)
                expected = expected if isinstance(expected, tuple) else (expected,)
                compiled = rebel.compile_from_torch(
                    model,
                    input_info=[("text", list(text.shape), "float32"), ("image", list(image.shape), "float32")],
                    npu=args.npu,
                    num_devices=1,
                )
                compiled.save(str(args.out / f"{label}.rbln"))
                runtime = compiled.create_runtime(tensor_type="pt", device=args.device)
                actual = runtime(text, image)
                actual = actual if isinstance(actual, (tuple, list)) else (actual,)
                actual = tuple(v.clone() for v in actual)
                repeated = runtime(text, image)
                repeated = repeated if isinstance(repeated, (tuple, list)) else (repeated,)
                torch.save(
                    {
                        "text": text,
                        "image": image,
                        "expected": expected,
                        "actual": actual,
                        "repeated": tuple(v.clone() for v in repeated),
                    },
                    args.out / f"{label}.pt",
                )
                row = {
                    "label": label,
                    "outputs": [
                        dict(metrics(a, e), repeat_equal=bool(torch.equal(a, r)))
                        for a, e, r in zip(actual, expected, repeated, strict=True)
                    ],
                }
                rows.append(row)
                dump_json(args.out / "head-controls.json", rows)
                print("SIGLIP2_HEAD " + json.dumps(row), flush=True)
                del runtime, compiled
                gc.collect()


def capture(args):
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    from optimum.rbln import RBLNSiglip2Model

    reference = AutoModel.from_pretrained(MODEL, revision=REVISION).float().eval()
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    fixture = hf_hub_download(
        "huggingface/documentation-images", "coco_sample.png", repo_type="dataset", revision=FIXTURE_REVISION
    )
    cat = Image.open(fixture).convert("RGB")
    images = [cat, cat.resize((256, 256)), cat.resize((128, 512)), cat.resize((512, 128))]
    texts = ["a photo of cats", "a photo of dogs", "a photo of a train", "a photo of parrots"]
    artifact = args.artifact or args.out / "original-b8"
    if args.artifact:
        model = RBLNSiglip2Model.from_pretrained(artifact, rbln_device=args.device)
    else:
        model = RBLNSiglip2Model.from_pretrained(
            MODEL, revision=REVISION, export=True, rbln_batch_size=8, rbln_npu=args.npu, rbln_device=args.device
        )
        model.save_pretrained(artifact)
    manifest = {
        "model": MODEL,
        "revision": REVISION,
        "fixture_revision": FIXTURE_REVISION,
        "fixture_sha256": hashlib.sha256(Path(fixture).read_bytes()).hexdigest(),
        "npu": args.npu,
        "artifact": str(artifact),
        "packages": {},
        "offsets": [],
    }
    manifest["compiled_config"] = {
        k: getattr(model.rbln_config, k) for k in ("batch_size", "text_batch_size", "max_seq_len", "max_patches")
    }
    assert manifest["compiled_config"] == {
        "batch_size": 8,
        "text_batch_size": 8,
        "max_seq_len": 64,
        "max_patches": 256,
    }
    try:
        manifest["checkout_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        manifest["checkout_diff_sha256"] = hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest()
    except (subprocess.CalledProcessError, FileNotFoundError):
        manifest["checkout_head"] = "unavailable"
    for package in ("rebel-compiler", "optimum-rbln", "transformers", "torch"):
        manifest["packages"][package] = importlib.metadata.version(package)
    dump_json(args.out / "manifest.json", manifest)
    first_inputs = first_actual = None
    for offset in range(4):
        inputs = processor(
            images=[images[(offset + i) % 4] for i in range(8)],
            text=[texts[(offset + i) % 4] for i in range(8)],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        expected = flatten(reference(**inputs))
        actual = flatten(model(**inputs))
        repeated = flatten(model(**inputs))
        case = args.out / f"original-b8-offset{offset}.pt"
        payload = {
            "inputs": dict(inputs),
            "expected": expected,
            "actual": actual,
            "repeated": repeated,
            "logit_scale": reference.logit_scale.detach().clone(),
            "logit_bias": reference.logit_bias.detach().clone(),
        }
        # Save every output before criteria checking or any potential diagnostic assertion.
        torch.save(payload, case)
        row = {"offset": offset, "inputs": {k: tensor_info(v) for k, v in inputs.items()}, "outputs": []}
        for name, a, e, r in zip(NAMES, actual, expected, repeated, strict=True):
            metric = dict(
                metrics(a, e),
                name=name,
                repeat_equal=bool(torch.equal(a, r)),
                actual=tensor_info(a),
                expected=tensor_info(e),
                repeated=tensor_info(r),
            )
            if name in ("text_embeds", "image_embeds"):
                metric["minimum_cosine"] = float(torch.nn.functional.cosine_similarity(a, e).min())
            checks = {"finite": metric["finite"], "nrmse": metric["nrmse"] <= 0.05, "repeat": metric["repeat_equal"]}
            if "pcc" in metric:
                checks["pcc"] = metric["pcc"] >= 0.99
            if "minimum_cosine" in metric:
                checks["minimum_cosine"] = metric["minimum_cosine"] >= 0.99
            metric["failed_criteria"] = [k for k, v in checks.items() if not v]
            row["outputs"].append(metric)
        manifest["offsets"].append(row)
        dump_json(args.out / "manifest.json", manifest)
        analyze(case)
        print("SIGLIP2_ORIGINAL " + json.dumps(row), flush=True)
        if first_inputs is None:
            first_inputs, first_actual = inputs, actual
    returned = flatten(model(**first_inputs))
    manifest["return_to_first_equal"] = [torch.equal(a, e) for a, e in zip(returned, first_actual, strict=True)]
    torch.save(returned, args.out / "return-to-first.pt")
    del model
    gc.collect()
    reloaded = RBLNSiglip2Model.from_pretrained(artifact, rbln_device=args.device)
    reload_actual = flatten(reloaded(**first_inputs))
    torch.save(reload_actual, args.out / "reloaded.pt")
    manifest["reload_equal"] = [torch.equal(a, e) for a, e in zip(reload_actual, first_actual, strict=True)]
    dump_json(args.out / "manifest.json", manifest)
    del reloaded, reference
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--npu", choices=["RBLN-CR13", "RBLN-CA25"], default="RBLN-CR13")
    parser.add_argument("--device", type=int)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--case", type=Path, help="Reuse captured case; skips full model")
    parser.add_argument("--analyze", type=Path, help="CPU-only existing capture analysis")
    parser.add_argument("--batches", type=int, nargs="+", default=[8])
    parser.add_argument("--skip-controls", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    with torch.no_grad():
        if args.analyze:
            print(json.dumps(analyze(args.analyze), indent=2))
            return
        if args.device is None:
            parser.error("--device must be supplied by the scheduler")
        assert importlib.metadata.version("rebel-compiler") == "0.12.1.dev328+g970308c6"
        args.out.mkdir(parents=True, exist_ok=True)
        if not args.case:
            capture(args)
        if not args.skip_controls:
            head_controls(args, args.case or args.out / "original-b8-offset0.pt")
        failures = []
        if not args.case:
            manifest = json.loads((args.out / "manifest.json").read_text())
            failures = [
                metric for case in manifest["offsets"] for metric in case["outputs"] if metric["failed_criteria"]
            ]
            assert all(manifest["return_to_first_equal"]), manifest["return_to_first_equal"]
            assert all(manifest["reload_equal"]), manifest["reload_equal"]
        print(
            "SIGLIP2_DIAGNOSTIC_COMPLETE "
            + json.dumps(
                {
                    "diagnostic_complete": True,
                    "original_failed_outputs": len(failures),
                    "product_acceptance": "requires_original_pytest",
                    "results": str(args.out),
                }
            ),
            flush=True,
        )
        assert not failures, failures


if __name__ == "__main__":
    main()
