import argparse
import os

from optimum.rbln import RBLNStableDiffusionPipeline


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id Lykon/dreamshaper-7 -o dreamshaper-7 \
#       --lora_ids latent-consistency/lcm-lora-sdv1-5 --guidance_scale 0.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Lykon/dreamshaper-7")
    parser.add_argument("--prompt", default="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k")
    parser.add_argument("--steps", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.model_id):
        pipe = RBLNStableDiffusionPipeline.from_pretrained(args.model_id)
    else:
        pipe = RBLNStableDiffusionPipeline.from_pretrained(
            args.model_id,
            lora_ids="latent-consistency/lcm-lora-sdv1-5",
            rbln_guidance_scale=0.0,
        )

    image = pipe(args.prompt, num_inference_steps=args.steps, guidance_scale=0).images[0]

    image.save(f"{args.prompt}_{args.steps}.png")


if __name__ == "__main__":
    main()
