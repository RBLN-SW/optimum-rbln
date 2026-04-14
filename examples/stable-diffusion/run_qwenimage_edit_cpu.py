"""
QwenImageEditPlus pipeline — CPU reference test (no RBLN compilation).

Usage:
    python run_qwenimage_edit_cpu.py
    python run_qwenimage_edit_cpu.py --model-id ./models --image-path assets/1.png
"""

import os

import fire
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline


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
    image_path: str = "/mnt/shared_data/.cache/pretrained_models/NC/VARCO_I2I_inference/assets/1.png",
    prompt: str = PROMPT,
    negative_prompt: str = " ",
    num_inference_steps: int = 40,
    true_cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,
    num_images_per_prompt: int = 1,
    seed: int = 0,
):
    pipe = QwenImageEditPlusPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=None)

    image = Image.open(image_path).convert("RGB").resize((1024, 1024))

    with torch.inference_mode():
        result = pipe(
            image=[image],
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=torch.manual_seed(seed),
            height=1024,
            width=1024,
        ).images[0]

    os.makedirs("results", exist_ok=True)
    output_name = "results/1_golden_output.png"
    result.save(output_name)
    print(f"Saved: {output_name}  ({result.size[0]}x{result.size[1]})")


if __name__ == "__main__":
    fire.Fire(main)
