import argparse
import os

import torch
from diffusers.utils import load_image

from optimum.rbln import RBLNKandinskyV22Img2ImgCombinedPipeline


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id kandinsky-community/kandinsky-2-2-decoder -o kandinsky-2-2-decoder \
#       --img_height 768 --img_width 768


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="kandinsky-community/kandinsky-2-2-decoder")
    parser.add_argument("--prompt", default="A red cartoon frog, 4k")
    return parser.parse_args()


def main():
    args = parse_args()

    img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/frog.png"
    init_image = load_image(img_url)

    if os.path.isdir(args.model_id):
        pipe = RBLNKandinskyV22Img2ImgCombinedPipeline.from_pretrained(args.model_id)
    else:
        pipe = RBLNKandinskyV22Img2ImgCombinedPipeline.from_pretrained(
            args.model_id,
            rbln_img_height=768,
            rbln_img_width=768,
        )

    generator = torch.manual_seed(42)

    image = pipe(
        prompt=args.prompt,
        image=init_image,
        height=768,
        width=768,
        num_inference_steps=100,
        strength=0.2,
        generator=generator,
    ).images[0]
    image.save(f"{args.prompt}.png")


if __name__ == "__main__":
    main()
