import argparse
import os

import torch
from PIL import Image

from optimum.rbln import RBLNQwenImageEditPlusPipeline


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen-Image-Edit",
        help="(str) model id or local diffusers checkpoint of the QwenImageEditPlus pipeline",
    )
    parser.add_argument("--image", type=str, required=True, help="(str) path to the RGB condition image")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Turn this sketch into a photorealistic product render on a white background.",
        help="(str) edit instruction",
    )
    parser.add_argument("--negative_prompt", type=str, default=" ")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--prompt_embed_length", type=int, default=1024, help="(int) compiled text-token length")
    parser.add_argument(
        "--visual_max_seq_len",
        type=int,
        default=2048,
        help="(int) vision-encoder max patches; 384x384 condition images need ~1600. Multiple of 64.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--from_diffusers",
        action="store_true",
        help="compile from the diffusers checkpoint and save to ./rbln_<model>; otherwise load a compiled dir",
    )
    parser.add_argument(
        "--create_runtimes",
        action="store_true",
        help="also allocate NPU runtimes right after compiling. All five submodules do not fit one "
        "140 GiB RBLN-CR13 together, so the default is compile-only; load them in stages for inference.",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    save_dir = "rbln_" + os.path.basename(os.path.normpath(args.model_id))

    if args.from_diffusers:
        # bfloat16 throughout: an fp32 build allocates ~129 GiB of runtimes across the five
        # submodules, which does not fit a 140 GiB device once the driver has its share.
        pipe = RBLNQwenImageEditPlusPipeline.from_pretrained(
            model_id=args.model_id,
            export=True,
            rbln_config={
                "text_encoder": {
                    "max_seq_len": 4096,
                    "visual": {"max_seq_len": args.visual_max_seq_len, "create_runtimes": args.create_runtimes},
                    "create_runtimes": args.create_runtimes,
                },
                "transformer": {
                    "batch_size": 1,
                    "prompt_embed_length": args.prompt_embed_length,
                    "create_runtimes": args.create_runtimes,
                },
                "vae": {"batch_size": 1, "create_runtimes": args.create_runtimes},
                "height": args.height,
                "width": args.width,
                "create_runtimes": args.create_runtimes,
            },
            torch_dtype=torch.bfloat16,
        )
        pipe.save_pretrained(save_dir)
        if not args.create_runtimes:
            print(f"compiled to {save_dir}; rerun without --from_diffusers to generate")
            return
    else:
        pipe = RBLNQwenImageEditPlusPipeline.from_pretrained(model_id=save_dir, export=False)

    image = Image.open(args.image).convert("RGB")
    result = pipe(
        image=image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    ).images[0]
    result.save("edited_image.png")
    print("saved edited_image.png")


if __name__ == "__main__":
    main()
