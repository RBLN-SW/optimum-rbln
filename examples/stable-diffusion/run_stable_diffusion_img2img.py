import argparse
import os
from io import BytesIO

import requests
import torch
from PIL import Image

from optimum.rbln import RBLNStableDiffusionImg2ImgPipeline


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id runwayml/stable-diffusion-v1-5 -o stable-diffusion-v1-5 \
#       --img_width 768 --img_height 512 --unet.batch_size 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt", default="A fantasy landscape, trending on artstation")
    parser.add_argument("--img-width", type=int, default=768)
    parser.add_argument("--img-height", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.75)
    return parser.parse_args()


def main():
    args = parse_args()

    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((args.img_width, args.img_height))

    if os.path.isdir(args.model_id):
        pipe = RBLNStableDiffusionImg2ImgPipeline.from_pretrained(args.model_id)
    else:
        pipe = RBLNStableDiffusionImg2ImgPipeline.from_pretrained(
            args.model_id,
            rbln_img_width=args.img_width,
            rbln_img_height=args.img_height,
            rbln_config={"unet": {"batch_size": 2}},
        )

    image = pipe(
        prompt=args.prompt,
        image=init_image,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        generator=torch.manual_seed(42),
    ).images[0]

    image.save(f"{args.prompt}.png")


if __name__ == "__main__":
    main()
