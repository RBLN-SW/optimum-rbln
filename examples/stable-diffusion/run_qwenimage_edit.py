"""
QwenImageEditPlus pipeline — compile & inference example.

Usage:
    # Compile from local model and save
    python run_qwenimage_edit.py --from-diffusers

    # Load already-compiled model and run inference
    python run_qwenimage_edit.py

    # Custom image size
    python run_qwenimage_edit.py --from-diffusers --height 1024 --width 1024
"""

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


def main(
    model_id: str = "/mnt/shared_data/.cache/pretrained_models/NC/VARCO_I2I_inference/models",
    from_diffusers: bool = False,
    image_path: str = "/mnt/shared_data/.cache/pretrained_models/NC/VARCO_I2I_inference/assets/1.png",
    prompt: str = PROMPT,
    negative_prompt: str = " ",
    height: int = None,
    width: int = None,
    num_inference_steps: int = 40,
    true_cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,
    num_images_per_prompt: int = 1,
    max_seq_len: int = 4096,
    prompt_embed_length: int = 512,
    seed: int = 0,
):
    save_dir = "rbln_" + os.path.basename(model_id)

    if from_diffusers:
        pipe = RBLNQwenImageEditPlusPipeline.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_config={
                "text_encoder": {
                    "max_seq_len": max_seq_len,
                    "visual": {
                        "max_seq_lens": 6400,
                    },
                },
                "transformer": {
                    "batch_size": 1,
                    "prompt_embed_length": prompt_embed_length,
                },
                "vae": {
                    "batch_size": 1,
                },
                **({"height": height, "width": width} if height and width else {}),
            },
        )
        pipe.save_pretrained(save_dir)
    else:
        pipe = RBLNQwenImageEditPlusPipeline.from_pretrained(
            model_id=save_dir,
            export=False,
        )

    image = Image.open(image_path).convert("RGB")

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
