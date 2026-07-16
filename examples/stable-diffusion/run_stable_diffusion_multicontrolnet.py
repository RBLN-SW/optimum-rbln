import argparse
import os

import cv2
import numpy as np
import torch
from controlnet_aux import OpenposeDetector
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image

from optimum.rbln import RBLNStableDiffusionControlNetPipeline


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --diffusion-model-id:
#
#   optimum-rbln-cli --model-id runwayml/stable-diffusion-v1-5 -o stable-diffusion-v1-5 \
#       --img_width 512 --img_height 512 --unet.batch_size 1 --controlnet.batch_size 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet-model-id", nargs="+", default=None)
    parser.add_argument("--prompt", default="a giant standing in a fantasy landscape, best quality")
    parser.add_argument(
        "--negative-prompt", default="monochrome, lowres, bad anatomy, worst quality, low quality"
    )
    return parser.parse_args()


def main():
    args = parse_args()

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

    controlnet_model_id = args.controlnet_model_id
    if controlnet_model_id is None:
        controlnet_model_id = [
            "lllyasviel/sd-controlnet-openpose",
            "lllyasviel/sd-controlnet-canny",
        ]

    controlnets = []
    for cmi in controlnet_model_id:
        controlnet = ControlNetModel.from_pretrained(cmi)
        controlnets.append(controlnet)

    if os.path.isdir(args.diffusion_model_id):
        pipe = RBLNStableDiffusionControlNetPipeline.from_pretrained(args.diffusion_model_id)
    else:
        pipe = RBLNStableDiffusionControlNetPipeline.from_pretrained(
            args.diffusion_model_id,
            controlnet=controlnets,
            rbln_img_width=512,
            rbln_img_height=512,
            rbln_config={
                "unet": {"batch_size": 1},
                "controlnet": {"batch_size": 1},
            },
            scheduler=UniPCMultistepScheduler.from_pretrained(args.diffusion_model_id, subfolder="scheduler"),
        )

    images = [openpose_image, canny_image]

    image = pipe(
        args.prompt,
        images,
        negative_prompt=args.negative_prompt,
        num_inference_steps=20,
        guidance_scale=0.0,
        controlnet_conditioning_scale=[1.0, 0.8],
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).images[0]

    image.save(f"{args.prompt}.jpg")


if __name__ == "__main__":
    main()
