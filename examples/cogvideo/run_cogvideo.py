"""Generate a video with CogVideoX-2b on RBLN NPUs.

    RBLN_DEVICES=0,1,2,3,4,5,6,7,8,9 python examples/cogvideo/run_cogvideo.py \
        --model-dir rbln-CogVideoX-2b --out-dir /path/to/new/output/folder

The first run compiles the text encoder, transformer and VAE (export=True) and saves them to
--model-dir; later runs load the compiled artifacts. The transformer runs with tensor parallel over
8 devices; the text encoder and the VAE get one device each (10 devices in total).

All submodules are compiled at --torch-dtype. The default is float16, the precision the CogVideoX-2b model card
recommends (the model was trained in FP16). The dtype is fixed at compile time: to change it, compile again into
a new --model-dir.
"""

import argparse
import os

import torch
from diffusers.utils import export_to_video

from optimum.rbln import RBLNCogVideoXPipeline


DEFAULT_PROMPT = (
    "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
    "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="THUDM/CogVideoX-2b")
    parser.add_argument("--model-dir", type=str, default="rbln-CogVideoX-2b")
    parser.add_argument("--out-dir", type=str, required=True, help="a new folder for the video and frames")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    # text_encoder: 1 device, transformer: tensor parallel over 8 devices, vae: 1 device (10 in total).
    devices = {"text_encoder": {"device": 0}, "transformer": {"device": list(range(1, 9))}, "vae": {"device": 9}}
    if not os.path.exists(os.path.join(args.model_dir, "model_index.json")):
        pipe = RBLNCogVideoXPipeline.from_pretrained(
            args.model_id,
            export=True,
            torch_dtype=getattr(torch, args.torch_dtype),
            rbln_config={"num_frames": args.num_frames, "guidance_scale": args.guidance_scale, **devices},
        )
        pipe.save_pretrained(args.model_dir)
    else:
        pipe = RBLNCogVideoXPipeline.from_pretrained(args.model_dir, export=False, rbln_config=devices)

    frames = pipe(
        prompt=args.prompt,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator().manual_seed(args.seed),
    ).frames[0]

    os.makedirs(args.out_dir, exist_ok=False)
    export_to_video(frames, os.path.join(args.out_dir, "video_rbln.mp4"), fps=8)
    os.makedirs(os.path.join(args.out_dir, "frames"))
    for i, frame in enumerate(frames):
        frame.save(os.path.join(args.out_dir, "frames", f"frame_{i:04d}.png"))
    print(f"saved {len(frames)} frames to {args.out_dir}")


if __name__ == "__main__":
    main()
