import argparse
import os

import torch
from diffusers.utils import load_image

from optimum.rbln import RBLNStableDiffusionInpaintPipeline


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id runwayml/stable-diffusion-inpainting -o stable-diffusion-inpainting \
#       --guidance_scale 7.5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument(
        "--prompt",
        default="concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
    mask_url = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
    )
    source = load_image(img_url)
    mask = load_image(mask_url)

    if os.path.isdir(args.model_id):
        pipe = RBLNStableDiffusionInpaintPipeline.from_pretrained(args.model_id)
    else:
        pipe = RBLNStableDiffusionInpaintPipeline.from_pretrained(
            args.model_id,
            rbln_guidance_scale=7.5,
        )

    image = pipe(args.prompt, image=source, mask_image=mask, generator=torch.manual_seed(42)).images[0]
    image.save(f"{args.prompt}.png")


if __name__ == "__main__":
    main()
