import argparse
from pathlib import Path

import torch
from diffusers.utils import export_to_video, load_video

from optimum.rbln import RBLNKandinsky6SRPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers")
    parser.add_argument("--video", required=True, help="Input video path or URL.")
    parser.add_argument("--compiled-model-path", default="kandinsky6-sr")
    parser.add_argument("--resolution-scale", type=float, choices=(2, 2.25, 4), default=2.25)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--output", default="kandinsky6-sr.mp4")
    return parser.parse_args()


def main():
    args = parse_args()
    video = load_video(args.video)[: args.num_frames]
    if Path(args.compiled_model_path, "model_index.json").is_file():
        pipe = RBLNKandinsky6SRPipeline.from_pretrained(args.compiled_model_path, rbln_device=args.device)
    else:
        pipe = RBLNKandinsky6SRPipeline.from_pretrained(
            args.model_id,
            export=True,
            model_save_dir=args.compiled_model_path,
            dtype=torch.bfloat16,
            rbln_device=args.device,
            rbln_config={
                "height": video[0].height,
                "width": video[0].width,
                "num_frames": len(video),
                "resolution_scale": args.resolution_scale,
            },
        )
    output = pipe(
        video=video,
        resolution_scale=args.resolution_scale,
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(args.seed),
    )
    export_to_video(output.frames[0], args.output, fps=args.fps)


if __name__ == "__main__":
    main()
