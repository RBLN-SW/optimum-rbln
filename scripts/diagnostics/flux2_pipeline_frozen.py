"""Native1024/prompt512 real-weight full FLUX2 klein reference-image edit pipeline acceptance on scheduled CR13."""

import argparse
import importlib
import importlib.metadata
import json
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.diagnostics.flux2_transformer_frozen import metrics, require_scheduled_execution, sha256


SDK = "0.12.1.dev333+gf66b88a7"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
MANIFEST_SHA = "9f6ec216df3e173ef8a0c8192d650daf13e580da0ada55ff8c5f022cc4dfc108"


def verify_bundle(directory):
    directory = Path(directory)
    if sha256(directory / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Full native pipeline manifest differs")
    manifest = json.loads((directory / "manifest.json").read_text())
    files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if files != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Frozen pipeline contains missing or unexpected files")
    for name, receipt in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Frozen bundle requires ordinary files inside its root")
        if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
            raise ValueError(f"Frozen file content differs: {name}")
    expected = {
        "revision": REVISION,
        "sdk": SDK,
        "target": "RBLN-CR13",
        "num_devices": 1,
        "batch": 1,
        "image_size": [1024, 1024],
        "prompt_length": 512,
        "reference_image_count": 1,
        "denoising_steps": 4,
        "guidance": 1.0,
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Native acceptance profile differs")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("debug-flux2-pipeline"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"phase": "preflight", "full_pipeline_npu_acceptance": False}
    started = time.monotonic()
    code = 1
    try:
        if args.execute:
            require_scheduled_execution()
        manifest = verify_bundle(args.artifacts)
        result["manifest_sha256"] = MANIFEST_SHA
        if args.verify_only:
            result["phase"] = "verified_only"
            code = 0
        else:
            import rebel
            import torch
            from tvm._ffi import base as ffi_base

            from optimum.rbln import (
                RBLNAutoencoderKLFlux2,
                RBLNFlux2KleinPipeline,
                RBLNFlux2KleinTextEncoder,
                RBLNFlux2Transformer2DModel,
            )

            if importlib.metadata.version("rebel-compiler") != SDK:
                raise ValueError("Runtime SDK differs from compiled artifacts")
            if sha256(Path(ffi_base._LIB._name).resolve()) != manifest["compiler_dso_sha256"]:
                raise ValueError("Loaded SDK library differs")
            for name, cls in [
                ("text_encoder", RBLNFlux2KleinTextEncoder),
                ("transformer", RBLNFlux2Transformer2DModel),
                ("vae", RBLNAutoencoderKLFlux2),
                ("pipeline", RBLNFlux2KleinPipeline),
            ]:
                module = importlib.import_module(cls.__module__)
                if sha256(Path(module.__file__)) != manifest["source_sha256"][name]:
                    raise ValueError(f"Runtime component source differs: {name}")
            if rebel.get_npu_name(0) != "RBLN-CR13":
                raise ValueError("Physical device differs from compiled CR13 target")
            golden = torch.load(args.artifacts / "golden.pt", weights_only=True)
            text = torch.load(args.artifacts / "text-golden.pt", weights_only=True)
            model = RBLNFlux2KleinPipeline.from_pretrained(
                args.artifacts / "model",
                export=False,
                rbln_create_runtimes=True,
                rbln_device=0,
                local_files_only=True,
            )
            result.update(phase="execute", sdk=SDK, placement=manifest["placement"], rows=[])
            rows = result["rows"]
            actuals = {}

            def save():
                torch.save(actuals, args.output / "actual.pt")
                (args.output / "progress.json").write_text(json.dumps(result, indent=2) + "\n")

            def compare(name, actual, expected, required=True):
                actuals[name] = actual.detach().clone()
                save()
                row = {"name": name, "required": required, **metrics(actual, expected)}
                rows.append(row)
                print(json.dumps(row), flush=True)

            with torch.inference_mode():
                hidden = model.text_encoder(**text["inputs"], output_hidden_states=True, use_cache=False).hidden_states
                if len(hidden) != 37:
                    raise ValueError("Text encoder must preserve every native hidden-state index")
                for index in (9, 18, 27):
                    compare(f"text_layer_{index}", hidden[index], text["hidden_states"][index])
                embeds, _ = model.encode_prompt(prompt=golden["prompt"], max_sequence_length=512)
                compare("full_prompt_embeddings", embeds, golden["denoiser"][0]["encoder_hidden_states"])
                # Teacher-forced decoder control keeps the component error separate
                # from accumulated denoising error in the full pipeline below.
                latents = model._unpack_latents_with_ids(
                    golden["step_latents"][-1],
                    golden["denoiser"][0]["img_ids"][:, : golden["step_latents"][-1].shape[1]],
                )
                mean = model.vae.bn.running_mean.view(1, -1, 1, 1)
                std = (model.vae.bn.running_var.view(1, -1, 1, 1) + model.vae.config.batch_norm_eps).sqrt()
                decoder_input = model._unpatchify_latents(latents * std + mean)
                compare(
                    "teacher_forced_decoder_raw",
                    model.vae.decode(decoder_input, return_dict=False)[0],
                    golden["decoder_raw"],
                )
                raw = []
                original_decode = model.vae.decode

                def capture_decode(*inputs, **kwargs):
                    output = original_decode(*inputs, **kwargs)
                    raw.append(
                        output[0].detach().clone() if isinstance(output, tuple) else output.sample.detach().clone()
                    )
                    return output

                model.vae.decode = capture_decode

                def callback(pipe, index, timestep, values):
                    compare(f"step_{index}_latents", values["latents"], golden["step_latents"][index])
                    return values

                kwargs = {
                    "prompt": golden["prompt"],
                    "num_inference_steps": 4,
                    "guidance_scale": 1.0,
                    "latents": golden["initial_latents"].clone(),
                    "max_sequence_length": 512,
                    "output_type": "pt",
                }
                from PIL import Image

                with Image.open(args.artifacts / "reference.png") as image:
                    reference = image.convert("RGB")
                import hashlib

                pixel_sha = hashlib.sha256(reference.tobytes()).hexdigest()
                if pixel_sha != manifest["reference_pixel_sha256"] or pixel_sha != golden["reference_pixel_sha256"]:
                    raise ValueError("Reference image pixels differ from native CPU input")
                if reference.size != (1024, 1024):
                    raise ValueError("Original reference geometry must remain 1024 square")
                kwargs["image"] = reference
                output = model(**kwargs, callback_on_step_end=callback).images
                compare("full_pipeline_decoder_raw", raw[0], golden["decoder_raw"])
                compare("full_pipeline_images", output, golden["images"])
                repeated = model(**kwargs, return_dict=False)[0]
                result["repeat_and_return_tuple_exact"] = torch.equal(repeated, output) and torch.equal(raw[1], raw[0])
                reloaded = RBLNFlux2KleinPipeline.from_pretrained(
                    args.artifacts / "model",
                    export=False,
                    rbln_create_runtimes=True,
                    rbln_device=0,
                    local_files_only=True,
                )
                if reloaded is model or reloaded.transformer.model[0] is model.transformer.model[0]:
                    raise ValueError("Reload requires independent runtime wrappers")
                reloaded_output = reloaded(**kwargs).images
                actuals["reloaded_images"] = reloaded_output
                result["reload_exact"] = torch.equal(reloaded_output, output)
                save()
            result["full_pipeline_npu_acceptance"] = (
                all(row["passed"] for row in rows if row["required"])
                and result["repeat_and_return_tuple_exact"]
                and result["reload_exact"]
            )
            result["phase"] = "finished"
            code = 0 if result["full_pipeline_npu_acceptance"] else 1
    except BaseException:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        result.update(exit_code=code, seconds=time.monotonic() - started)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        suite = ET.Element("testsuite", name="flux2_full_pipeline_native1024", tests="1", failures=str(int(code != 0)))
        case = ET.SubElement(suite, "testcase", name=result["phase"])
        if code:
            ET.SubElement(case, "failure").text = result.get("error", "Original numerical or state criteria failed")
        ET.ElementTree(suite).write(args.output / "junit.xml", encoding="unicode")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
