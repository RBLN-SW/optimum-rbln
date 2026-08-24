import argparse
import os

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image

from optimum.rbln import RBLNStableDiffusionControlNetPipeline


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --diffusion-model-id:
#
#   optimum-rbln-cli --model-id runwayml/stable-diffusion-v1-5 -o stable-diffusion-v1-5 \
#       --unet.batch_size 2 --controlnet.batch_size 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet-model-id", default="lllyasviel/sd-controlnet-canny")
    parser.add_argument("--prompt", default="the mona lisa")
    return parser.parse_args()


def main():
    args = parse_args()

    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_id)

    image = load_image(
        "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    )

    # input image preprocessing
    np_image = np.array(image)
    np_image = cv2.Canny(np_image, 100, 200)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    canny_image = Image.fromarray(np_image)

    img_width, img_height = image.size

    if os.path.isdir(args.diffusion_model_id):
        pipe = RBLNStableDiffusionControlNetPipeline.from_pretrained(args.diffusion_model_id)
    else:
        pipe = RBLNStableDiffusionControlNetPipeline.from_pretrained(
            args.diffusion_model_id,
            controlnet=controlnet,
            rbln_img_width=img_width,
            rbln_img_height=img_height,
            rbln_config={
                "unet": {"batch_size": 2},
                "controlnet": {"batch_size": 2},
            },
            scheduler=UniPCMultistepScheduler.from_pretrained(args.diffusion_model_id, subfolder="scheduler"),
        )

    image = pipe(
        prompt=args.prompt,
        image=canny_image,
        generator=torch.manual_seed(42),
    ).images[0]

    image.save(f"{args.prompt}.png")


if __name__ == "__main__":
    main()
