import os
from typing import List, Optional

import cv2
import fire
import numpy as np
import torch
from controlnet_aux import OpenposeDetector
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image

from optimum.rbln import RBLNStableDiffusionControlNetPipeline


def main(
    diffusion_model_id: str = "runwayml/stable-diffusion-v1-5",
    from_diffusers: bool = False,
    controlnet_model_id: Optional[List[str]] = None,
    prompt: str = "a giant standing in a fantasy landscape, best quality",
    negative_prompt: str = "monochrome, lowres, bad anatomy, worst quality, low quality",
):
    canny_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
    )
    canny_image = np.array(canny_image)

    low_threshold = 100
    high_threshold = 200

    canny_image = cv2.Canny(canny_image, low_threshold, high_threshold)
    zero_start = canny_image.shape[1] // 4
    zero_end = zero_start + canny_image.shape[1] // 2
    canny_image[:, zero_start:zero_end] = 0

    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    openpose_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"
    )
    openpose_image = openpose(openpose_image)

    if controlnet_model_id is None:
        controlnet_model_id = [
            "lllyasviel/sd-controlnet-openpose",
            "lllyasviel/sd-controlnet-canny",
        ]

    controlnets = []
    for cmi in controlnet_model_id:
        controlnet = ControlNetModel.from_pretrained(cmi)
        controlnets.append(controlnet)

    if from_diffusers:
        pipe = RBLNStableDiffusionControlNetPipeline.from_pretrained(
            model_id=diffusion_model_id,
            controlnet=controlnets,
            rbln_img_width=512,
            rbln_img_height=512,
            export=True,
            scheduler=UniPCMultistepScheduler.from_pretrained(diffusion_model_id, subfolder="scheduler"),
        )
        pipe.save_pretrained(os.path.basename(diffusion_model_id))
    else:
        pipe = RBLNStableDiffusionControlNetPipeline.from_pretrained(
            model_id=os.path.basename(diffusion_model_id),
            export=False,
        )

    images = [openpose_image, canny_image]

    image = pipe(
        prompt,
        images,
        negative_prompt=negative_prompt,
        num_inference_steps=20,
        guidance_scale=0.0,
        controlnet_conditioning_scale=[1.0, 0.8],
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).images[0]

    image.save(f"{prompt}.jpg")


if __name__ == "__main__":
    fire.Fire(main)
