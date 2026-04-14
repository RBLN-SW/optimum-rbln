"""
QwenImageEditPlus pipeline — compile & inference example.

Usage:
    # Compile from local model and save
    python run_qwenimage_edit.py --from-diffusers

    # Load already-compiled model and run inference
    python run_qwenimage_edit.py

    # Custom image size
    python run_qwenimage_edit.py --from-diffusers --height 1024 --width 1024

    # Quick test with reduced layers (fast compile, garbage output)
    python run_qwenimage_edit.py --from-diffusers --transformer-num-layers 1 --text-encoder-num-layers 1 --vision-depth 2
"""

import json
import os

import fire
import torch
from PIL import Image

from optimum.rbln import RBLNQwenImageEditPlusPipeline


PROMPT = """\
Task:
Transform the provided engineering blueprint into a realistic 3D mechanical model rendering.

Instructions:
Interpret the technical drawing as an assembly blueprint and reconstruct the complete mechanical device as a single assembled 3D object.

Requirements:
Use the blueprint as the structural reference.
Ignore the original color and texture of the input.
Convert the orthographic views and cross-section diagrams into a coherent 3D form.
Preserve the mechanical relationships between components such as gears, shafts, bearings, and housing.
Reconstruct the device as a fully assembled object (not exploded view).
Remove all blueprint elements such as dimension lines, labels, arrows, part numbers, and annotations.

Output style:
Photorealistic industrial CAD rendering
Clean studio lighting
Neutral gray background
Render the object as uniform brushed silver metal (stainless steel or aluminum)
High precision mechanical surfaces
Visible internal structure via cutaway section (similar to a professional product visualization)

Composition:
Single centered object
Slight perspective view
High detail mechanical engineering visualization
Physically plausible geometry and proportions

Important:
Do not produce a schematic drawing or blueprint.
The output must look like a real manufactured mechanical component rendered in 3D.
"""


def _load_thin_submodules(model_id, transformer_num_layers, text_encoder_num_layers, vision_depth):
    """Load submodules with reduced layers for fast compile testing."""
    submodules = {}

    if transformer_num_layers is not None:
        from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

        print(f"  Loading transformer with num_layers={transformer_num_layers}")
        submodules["transformer"] = QwenImageTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer", num_layers=transformer_num_layers,
        )

    if text_encoder_num_layers is not None or vision_depth is not None:
        from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

        config = AutoConfig.from_pretrained(model_id, subfolder="text_encoder")
        if text_encoder_num_layers is not None:
            text_config = json.loads(config.text_config.to_json_string())
            text_config["num_hidden_layers"] = text_encoder_num_layers
            text_config["layer_types"] = text_config.get("layer_types", ["dense"] * 28)[:text_encoder_num_layers]
            config.text_config = type(config.text_config)(**text_config)
            config.num_hidden_layers = text_encoder_num_layers
            print(f"  Loading text_encoder with num_hidden_layers={text_encoder_num_layers}")
        if vision_depth is not None:
            vision_config = json.loads(config.vision_config.to_json_string())
            vision_config["depth"] = vision_depth
            vision_config["fullatt_block_indexes"] = [i for i in vision_config.get("fullatt_block_indexes", []) if i < vision_depth]
            if not vision_config["fullatt_block_indexes"]:
                vision_config["fullatt_block_indexes"] = [vision_depth - 1]
            config.vision_config = type(config.vision_config)(**vision_config)
            print(f"  Loading text_encoder.visual with depth={vision_depth}")

        submodules["text_encoder"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, subfolder="text_encoder", config=config,
        )

    return submodules


def main(
    model_id: str = "/mnt/shared_data/.cache/pretrained_models/NC/VARCO_I2I_inference/models",
    from_diffusers: bool = False,
    image_path: str = "/mnt/shared_data/.cache/pretrained_models/NC/VARCO_I2I_inference/assets/1.png",
    prompt: str = PROMPT,
    negative_prompt: str = " ",
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 40,
    true_cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,
    num_images_per_prompt: int = 1,
    max_seq_len: int = 4096,
    prompt_embed_length: int = 512,
    seed: int = 0,
    transformer_num_layers: int = None,
    text_encoder_num_layers: int = None,
    vision_depth: int = None,
):
    save_dir = "rbln_" + os.path.basename(model_id)

    if from_diffusers:
        thin = transformer_num_layers is not None or text_encoder_num_layers is not None or vision_depth is not None
        extra_kwargs = {}
        if thin:
            print("\n[Thin mode] Loading submodules with reduced layers...")
            extra_kwargs = _load_thin_submodules(
                model_id, transformer_num_layers, text_encoder_num_layers, vision_depth,
            )

        pipe = RBLNQwenImageEditPlusPipeline.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_config={
                "text_encoder": {
                    "max_seq_len": max_seq_len,
                    "visual": {
                        "max_seq_lens": 6400,
                        "device": [24]
                    },
                    "tensor_parallel_size": 4,
                    "device": [28, 29, 30, 31],
                },
                "transformer": {
                    "batch_size": 1,
                    "prompt_embed_length": prompt_embed_length,
                    "tensor_parallel_size": 8,
                    "device": [16, 17, 18, 19, 20, 21, 22, 23],
                },
                "vae": {
                    "batch_size": 1,
                    "device": [25],
                },
                **({"height": height, "width": width} if height and width else {}),
            },
            torch_dtype=torch.float32,
            **extra_kwargs,
        )
        pipe.save_pretrained(save_dir)
    else:
        pipe = RBLNQwenImageEditPlusPipeline.from_pretrained(
            model_id=save_dir,
            export=False,
            rbln_config={
                "text_encoder": {
                    "device": [28, 29, 30, 31],
                    "visual": {
                        "device": [24]
                    },
                },
                "transformer": {
                    "device": [16, 17, 18, 19, 20, 21, 22, 23],
                },
                "vae": {
                    "device": [25],
                },
            },
        )

    image = Image.open(image_path).convert("RGB").resize((height, width))

    result = pipe(
        image=[image],
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=torch.manual_seed(seed),
        **({"height": height, "width": width} if height and width else {}),
    ).images[0]

    os.makedirs("results", exist_ok=True)
    output_name = "results/1_output.png"
    result.save(output_name)
    print(f"Saved: {output_name}  ({result.size[0]}x{result.size[1]})")


if __name__ == "__main__":
    fire.Fire(main)
