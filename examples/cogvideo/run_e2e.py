"""CogVideoX-2b text-to-video end to end: CPU fp32 golden and RBLN, compared frame by frame.

1) CPU golden (diffusers CogVideoXPipeline, fp32 on the host):
    python examples/cogvideo/run_e2e.py cpu
2) RBLN: compile (export=True) and save, reload with export=False, generate with the same seed,
   compare with the golden video (per-frame PSNR and Pearson r):
    RBLN_DEVICES=0,1,2,3,4,5,6,7,8,9 python examples/cogvideo/run_e2e.py rbln --golden <golden dir> \
        --compiled-dir /home/seinpark/nas_data/cogvideox/compiled/pipeline_cogvideox_2b

Each run writes a new folder under OUT_ROOT (video, frames, latents, meta.json).
"""

import argparse
import json
import time
from pathlib import Path

import torch
from cog_utils import compare, new_out_dir, psnr, save_video_and_frames, write_meta


MODEL_ID = "THUDM/CogVideoX-2b"
PROMPT = (
    "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
    "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few "
    "other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall "
    "bamboo, casting a gentle glow on the scene."
)


def generate(pipe, args, out_dir: Path, tag: str) -> tuple[torch.Tensor, float]:
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    latents = pipe(
        prompt=PROMPT,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=gen,
        output_type="latent",
    ).frames
    gen_time = time.time() - t0
    torch.save(latents, out_dir / f"latents_{tag}.pt")
    with torch.no_grad():
        video = pipe.decode_latents(latents)  # (B, C, F, H, W) in [-1, 1]
    torch.save(video.half(), out_dir / f"video_{tag}.pt")
    save_video_and_frames(video, out_dir, tag)
    return video, gen_time


def compare_with_golden(video: torch.Tensor, latents: torch.Tensor, golden: Path) -> dict:
    ref = torch.load(golden / "video_cpu.pt").float()
    video = video.float()
    results = {"overall": compare("video overall", video, ref)}
    results["psnr_overall"] = psnr(video.clamp(-1, 1), ref.clamp(-1, 1))
    per_frame = []
    for f in range(ref.shape[2]):
        stats = compare(f"frame {f:02d}", video[:, :, f], ref[:, :, f])
        stats["psnr"] = psnr(video[:, :, f].clamp(-1, 1), ref[:, :, f].clamp(-1, 1))
        per_frame.append(stats)
    results["per_frame"] = per_frame
    print(f"overall PSNR {results['psnr_overall']:.2f} dB")
    print("per-frame PSNR(dB): " + " ".join(f"{s['psnr']:.1f}" for s in per_frame))
    print("per-frame r: " + " ".join(f"{s['pearson_r']:.4f}" for s in per_frame))
    results["latents"] = compare("final latents", latents, torch.load(golden / "latents_cpu.pt"))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["cpu", "rbln", "compare"])
    parser.add_argument("--rbln-dir", type=str, default=None, help="compare: folder of an RBLN run")
    parser.add_argument("--golden", type=str, default=None, help="folder of a CPU golden run")
    parser.add_argument("--compiled-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    meta = {
        "script": "examples/cogvideo/run_e2e.py",
        "mode": args.mode,
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "seed": args.seed,
        "num_frames": args.num_frames,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": 480,
        "width": 720,
    }

    if args.mode == "cpu":
        from diffusers import CogVideoXPipeline

        torch.set_num_threads(96)
        out_dir = new_out_dir("e2e_cpu_golden")
        pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        video, gen_time = generate(pipe, args, out_dir, "cpu")
        write_meta(out_dir, {**meta, "dtype": "torch.float32 (CPU)", "generate_time_s": gen_time, "notes": args.notes})
        print(f"CPU golden: {gen_time:.0f}s, saved to {out_dir}")
        return

    if args.mode == "compare":
        rbln_dir = Path(args.rbln_dir)
        video = torch.load(rbln_dir / "video_rbln.pt").float()
        results = compare_with_golden(video, torch.load(rbln_dir / "latents_rbln.pt"), Path(args.golden))
        meta_path = rbln_dir / "meta.json"
        saved = json.loads(meta_path.read_text())
        saved.update({"golden_dir": args.golden, **results})
        meta_path.write_text(json.dumps(saved, indent=2, default=str))
        print(f"updated {meta_path}")
        return

    from optimum.rbln import RBLNCogVideoXPipeline

    out_dir = new_out_dir("e2e")
    compiled_dir = Path(args.compiled_dir)
    t0 = time.time()
    if not (compiled_dir / "model_index.json").exists():
        pipe = RBLNCogVideoXPipeline.from_pretrained(
            MODEL_ID,
            export=True,
            rbln_config={
                "num_frames": args.num_frames,
                "guidance_scale": args.guidance_scale,
                "create_runtimes": False,
            },
        )
        pipe.save_pretrained(compiled_dir)
        del pipe
    compile_time = time.time() - t0
    print(f"compile + save: {compile_time:.0f}s -> {compiled_dir}")

    t0 = time.time()
    pipe = RBLNCogVideoXPipeline.from_pretrained(
        compiled_dir,
        export=False,
        rbln_config={
            "text_encoder": {"device": 0},
            "transformer": {"device": list(range(1, 9))},
            "vae": {"device": 9},
        },
    )
    print(f"load (export=False): {time.time() - t0:.1f}s")
    video, gen_time = generate(pipe, args, out_dir, "rbln")
    print(f"RBLN generate: {gen_time:.1f}s")

    results = {}
    if args.golden:
        results = compare_with_golden(video, torch.load(out_dir / "latents_rbln.pt"), Path(args.golden))

    write_meta(
        out_dir,
        {
            **meta,
            "dtype_rbln": "float32 graphs (text_encoder, transformer, vae)",
            "golden_dir": args.golden,
            "compiled_dir": str(compiled_dir),
            "compile_time_s": compile_time,
            "generate_time_s": gen_time,
            **results,
            "notes": args.notes,
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "psnr": results.get("psnr_overall")}))


if __name__ == "__main__":
    main()
