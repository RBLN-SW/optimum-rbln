import argparse
import os

import torch

from optimum.rbln import RBLNKandinskyV22Pipeline, RBLNKandinskyV22PriorPipeline


# You can compile the models ahead of time with the CLI and then load the
# artifacts here by passing the output directories as --prior-model-id / --inpaint-model-id:
#
#   optimum-rbln-cli --model-id kandinsky-community/kandinsky-2-2-prior -o kandinsky-2-2-prior
#   optimum-rbln-cli --model-id kandinsky-community/kandinsky-2-2-decoder -o kandinsky-2-2-decoder \
#       --img_height 768 --img_width 768


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-model-id", default="kandinsky-community/kandinsky-2-2-prior")
    parser.add_argument("--inpaint-model-id", default="kandinsky-community/kandinsky-2-2-decoder")
    parser.add_argument("--prompt", default="red cat, 4k photo")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.inpaint_model_id):
        prior_pipe = RBLNKandinskyV22PriorPipeline.from_pretrained(args.prior_model_id)
        pipe = RBLNKandinskyV22Pipeline.from_pretrained(args.inpaint_model_id)
    else:
        prior_pipe = RBLNKandinskyV22PriorPipeline.from_pretrained(args.prior_model_id)
        pipe = RBLNKandinskyV22Pipeline.from_pretrained(
            args.inpaint_model_id,
            rbln_img_height=768,
            rbln_img_width=768,
        )

    generator = torch.manual_seed(42)
    out = prior_pipe(args.prompt, generator=generator)
    image_emb = out.image_embeds
    zero_image_emb = out.negative_image_embeds

    out = pipe(
        image_embeds=image_emb,
        negative_image_embeds=zero_image_emb,
        height=768,
        width=768,
        num_inference_steps=50,
        generator=generator,
    )
    image = out.images[0]
    image.save(f"{args.prompt}.png")


if __name__ == "__main__":
    main()
