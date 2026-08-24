import argparse
import os

import torch

from optimum.rbln import RBLNStableDiffusionPipeline


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id runwayml/stable-diffusion-v1-5 -o stable-diffusion-v1-5 \
#       --unet.batch_size 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt", default="an illustration of a cute white cat riding a black horse on mars")
    parser.add_argument("--steps", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.model_id):
        pipe = RBLNStableDiffusionPipeline.from_pretrained(args.model_id)
    else:
        pipe = RBLNStableDiffusionPipeline.from_pretrained(
            args.model_id,
            rbln_config={"unet": {"batch_size": 2}},
        )

    image = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        generator=torch.manual_seed(42),
    ).images[0]
    image.save(f"{args.prompt}.png")


if __name__ == "__main__":
    main()
