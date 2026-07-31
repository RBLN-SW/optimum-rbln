import argparse
import os

import torch
from diffusers.utils import load_image

from optimum.rbln import RBLNKandinskyV22Img2ImgPipeline, RBLNKandinskyV22PriorPipeline


# You can compile the models ahead of time with the CLI and then load the
# artifacts here by passing the output directories as --prior-model-id / --inpaint-model-id:
#
#   optimum-rbln-cli --model-id kandinsky-community/kandinsky-2-2-prior -o kandinsky-2-2-prior \
#       --prior.batch_size 2
#   optimum-rbln-cli --model-id kandinsky-community/kandinsky-2-2-decoder -o kandinsky-2-2-decoder \
#       --img_height 768 --img_width 768 --unet.batch_size 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-model-id", default="kandinsky-community/kandinsky-2-2-prior")
    parser.add_argument("--inpaint-model-id", default="kandinsky-community/kandinsky-2-2-decoder")
    parser.add_argument("--prompt", default="A red cartoon frog, 4k")
    return parser.parse_args()


def main():
    args = parse_args()

    img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/frog.png"
    init_image = load_image(img_url)

    if os.path.isdir(args.inpaint_model_id):
        prior_pipe = RBLNKandinskyV22PriorPipeline.from_pretrained(args.prior_model_id)
        pipe = RBLNKandinskyV22Img2ImgPipeline.from_pretrained(args.inpaint_model_id)
    else:
        prior_pipe = RBLNKandinskyV22PriorPipeline.from_pretrained(
            args.prior_model_id,
            rbln_config={"prior": {"batch_size": 2}},
        )
        pipe = RBLNKandinskyV22Img2ImgPipeline.from_pretrained(
            args.inpaint_model_id,
            rbln_img_height=768,
            rbln_img_width=768,
            rbln_config={"unet": {"batch_size": 2}},
        )

    generator = torch.manual_seed(42)
    image_emb, zero_image_emb = prior_pipe(args.prompt, generator=generator, return_dict=False)

    out = pipe(
        image=init_image,
        image_embeds=image_emb,
        negative_image_embeds=zero_image_emb,
        height=768,
        width=768,
        num_inference_steps=100,
        strength=0.2,
        generator=generator,
    )
    image = out.images[0]
    image.save(f"{args.prompt}.png")


if __name__ == "__main__":
    main()
