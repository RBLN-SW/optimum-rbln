"""Compare the RBLN CogVideoX VAE decoder with the CPU fp32 `AutoencoderKLCogVideoX.decode`.

Example:
    RBLN_DEVICES=4 python examples/cogvideo/verify_vae.py --num-frames 49 \
        --compiled-dir /home/seinpark/nas_data/cogvideox/compiled/vae_f49_480x720

The script compiles the decoder (or loads it from --compiled-dir), decodes a fixed-seed latent on
both sides, prints Pearson r with the offset/scale/abs-error stats overall and per output frame,
and stores the videos, frames, latent and meta.json in a new folder under OUT_ROOT.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from cog_utils import (
    DeviceMemSampler,
    compare,
    host_peak_rss_gb,
    new_out_dir,
    psnr,
    save_video_and_frames,
    write_meta,
)
from diffusers import AutoencoderKLCogVideoX

from optimum.rbln import RBLNAutoencoderKLCogVideoX


MODEL_ID = "THUDM/CogVideoX-2b"


def print_alloc(compiled_dir: Path) -> None:
    import rebel

    for path in sorted(compiled_dir.glob("*.rbln")):
        alloc = rebel.RBLNCompiledModel(str(path)).get_alloc_per_node_by_key()
        gib = {key: round(sum(vals) / 2**30, 3) for key, vals in alloc.items() if sum(vals)}
        print(f"device alloc {path.name}: total={sum(gib.values()):.3f} GiB {gib}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=49, help="pixel frames the graphs are compiled for")
    parser.add_argument(
        "--decode-latent-frames", type=int, default=None, help="latent frames to decode (default: all compiled)"
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compiled-dir", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache-dtype", type=str, default=None)
    parser.add_argument("--num-devices", type=int, default=None, help="RSD: split each graph across N devices")
    parser.add_argument("--desc", type=str, default=None)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--cpu-ref", type=str, default=None, help="reuse a saved CPU reference (video_cpu.pt)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    vae_cpu = AutoencoderKLCogVideoX.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32).eval()
    num_latent = (args.num_frames - 1) // 4 + 1
    decode_latent = args.decode_latent_frames or num_latent
    latent_shape = (1, vae_cpu.config.latent_channels, decode_latent, args.height // 8, args.width // 8)
    gen = torch.Generator().manual_seed(args.seed)
    # Scaled like the pipeline does before decode (latents / scaling_factor).
    z = torch.randn(latent_shape, generator=gen) / vae_cpu.config.scaling_factor
    print(f"seed={args.seed} latent shape={tuple(z.shape)} (compiled num_frames={args.num_frames})")

    compiled_dir = Path(args.compiled_dir) if args.compiled_dir else None
    device = list(range(args.device, args.device + args.num_devices)) if args.num_devices else args.device
    compile_time = None
    rss_before = host_peak_rss_gb()
    with DeviceMemSampler() as mem:
        if compiled_dir is None or not (compiled_dir / "rbln_config.json").exists():
            # Compile without runtimes and save first, so the artifacts survive a failed device load.
            compiled_dir = compiled_dir or Path(new_out_dir("vae_compiled_tmp"))
            t0 = time.time()
            vae_compiled = RBLNAutoencoderKLCogVideoX.from_pretrained(
                MODEL_ID,
                subfolder="vae",
                export=True,
                rbln_config={
                    "num_frames": args.num_frames,
                    "height": args.height,
                    "width": args.width,
                    "create_runtimes": False,
                    **({"cache_dtype": args.cache_dtype} if args.cache_dtype else {}),
                    **({"num_devices": args.num_devices} if args.num_devices else {}),
                },
            )
            compile_time = time.time() - t0
            print(f"compiled VAE in {compile_time:.1f}s")
            vae_compiled.save_pretrained(compiled_dir)
            print(f"saved compiled VAE to {compiled_dir}")
            del vae_compiled
        print_alloc(compiled_dir)
        t0 = time.time()
        vae_rbln = RBLNAutoencoderKLCogVideoX.from_pretrained(
            compiled_dir, export=False, rbln_config={"device": device}
        )
        print(f"loaded compiled VAE from {compiled_dir} in {time.time() - t0:.1f}s")
        compile_rss = host_peak_rss_gb()

        t0 = time.time()
        with torch.no_grad():
            out_rbln = vae_rbln.decode(z).sample
        rbln_time = time.time() - t0
    print(f"RBLN decode {rbln_time:.2f}s  out shape={tuple(out_rbln.shape)} dtype={out_rbln.dtype}")
    print(f"host peak RSS: before load/compile {rss_before:.2f} GB, after {compile_rss:.2f} GB")
    print(f"device peak memory (GiB, rbln-stat): {mem.summary()}")

    if args.cpu_ref:
        out_cpu = torch.load(args.cpu_ref)
    else:
        t0 = time.time()
        with torch.no_grad():
            out_cpu = vae_cpu.decode(z).sample
        print(f"CPU decode {time.time() - t0:.1f}s")

    print(f"RBLN dtype: {vae_rbln.rbln_config.dtype} (compiled graphs), reference: CPU float32")
    overall = compare("overall", out_rbln, out_cpu)
    per_frame = []
    for f in range(out_cpu.shape[2]):
        stats = compare(f"frame {f:02d}", out_rbln[:, :, f], out_cpu[:, :, f])
        stats["psnr"] = psnr(out_rbln[:, :, f].clamp(-1, 1), out_cpu[:, :, f].clamp(-1, 1))
        per_frame.append(stats)
    print("per-frame r: " + " ".join(f"{s['pearson_r']:.4f}" for s in per_frame))
    print("per-frame PSNR(dB): " + " ".join(f"{s['psnr']:.1f}" for s in per_frame))

    desc = args.desc or f"vae_lat{decode_latent}_f{out_cpu.shape[2]}"
    out_dir = new_out_dir(desc)
    torch.save(z, out_dir / "latents.pt")
    if not args.cpu_ref:
        torch.save(out_cpu, out_dir / "video_cpu.pt")
    save_video_and_frames(out_cpu, out_dir, "cpu")
    save_video_and_frames(out_rbln, out_dir, "rbln")
    write_meta(
        out_dir,
        {
            "script": "examples/cogvideo/verify_vae.py",
            "model_id": MODEL_ID,
            "seed": args.seed,
            "latent_shape": list(z.shape),
            "compiled_num_frames": args.num_frames,
            "num_frames_decoded": out_cpu.shape[2],
            "height": args.height,
            "width": args.width,
            "dtype_rbln": str(vae_rbln.rbln_config.dtype),
            "dtype_reference": "torch.float32 (CPU)",
            "num_devices": args.num_devices,
            "compiled_dir": str(compiled_dir) if compiled_dir else None,
            "compile_time_s": compile_time,
            "rbln_decode_time_s": rbln_time,
            "host_peak_rss_gb": compile_rss,
            "device_peak_mem_gib": mem.summary(),
            "overall": overall,
            "psnr_overall": psnr(out_rbln.clamp(-1, 1), out_cpu.clamp(-1, 1)),
            "per_frame": per_frame,
            "notes": args.notes,
        },
    )
    print(f"saved outputs to {out_dir}")
    print(json.dumps({"r": overall["pearson_r"], "out_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
