import argparse
import os

import numpy as np
import torch
from diffusers.utils import load_image

from optimum.rbln import RBLNKandinskyV22InpaintCombinedPipeline


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id kandinsky-community/kandinsky-2-2-decoder-inpaint -o kandinsky-2-2-decoder-inpaint \
#       --img_height 768 --img_width 768


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="kandinsky-community/kandinsky-2-2-decoder-inpaint")
    parser.add_argument("--prompt", default="a hat")
    return parser.parse_args()


def main():
    args = parse_args()

    img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"
    init_image = load_image(img_url)

    if os.path.isdir(args.model_id):
        pipe = RBLNKandinskyV22InpaintCombinedPipeline.from_pretrained(args.model_id)
    else:
        pipe = RBLNKandinskyV22InpaintCombinedPipeline.from_pretrained(
            args.model_id,
            rbln_img_height=768,
            rbln_img_width=768,
        )

    generator = torch.manual_seed(42)
    # Mask out the desired area to inpaint
    # In this example, we will draw a hat on the cat's head
    mask = np.zeros((768, 768), dtype=np.float32)
    mask[:250, 250:-250] = 1

    image = pipe(args.prompt, image=init_image, mask_image=mask, generator=generator).images[0]
    image.save(f"{args.prompt}.png")


if __name__ == "__main__":
    main()
