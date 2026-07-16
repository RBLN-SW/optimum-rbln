import argparse
import os

import numpy as np
import torch
from diffusers.utils import load_image

from optimum.rbln import RBLNKandinskyV22InpaintPipeline, RBLNKandinskyV22PriorPipeline


# You can compile the models ahead of time with the CLI and then load the
# artifacts here by passing the output directories as --prior-model-id / --inpaint-model-id:
#
#   optimum-rbln-cli --model-id kandinsky-community/kandinsky-2-2-prior -o kandinsky-2-2-prior
#   optimum-rbln-cli --model-id kandinsky-community/kandinsky-2-2-decoder-inpaint -o kandinsky-2-2-decoder-inpaint \
#       --img_height 768 --img_width 768


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-model-id", default="kandinsky-community/kandinsky-2-2-prior")
    parser.add_argument("--inpaint-model-id", default="kandinsky-community/kandinsky-2-2-decoder-inpaint")
    parser.add_argument("--prompt", default="a hat")
    return parser.parse_args()


def main():
    args = parse_args()

    img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"
    init_image = load_image(img_url)

    if os.path.isdir(args.inpaint_model_id):
        prior_pipe = RBLNKandinskyV22PriorPipeline.from_pretrained(args.prior_model_id)
        pipe = RBLNKandinskyV22InpaintPipeline.from_pretrained(args.inpaint_model_id)
    else:
        prior_pipe = RBLNKandinskyV22PriorPipeline.from_pretrained(args.prior_model_id)
        pipe = RBLNKandinskyV22InpaintPipeline.from_pretrained(
            args.inpaint_model_id,
            rbln_img_width=768,
            rbln_img_height=768,
        )

    generator = torch.manual_seed(42)
    image_emb, zero_image_emb = prior_pipe(args.prompt, generator=generator, return_dict=False)

    # Mask out the desired area to inpaint
    # In this example, we will draw a hat on the cat's head
    mask = np.zeros((768, 768), dtype=np.float32)
    mask[:250, 250:-250] = 1

    out = pipe(
        image=init_image,
        mask_image=mask,
        image_embeds=image_emb,
        negative_image_embeds=zero_image_emb,
        generator=generator,
    )
    image = out.images[0]
    image.save(f"{args.prompt}.png")


if __name__ == "__main__":
    main()
