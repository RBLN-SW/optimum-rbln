"""Compare the RBLN CogVideoX VAE encoder (image-to-video, 1-frame image) with the CPU fp32 `vae.encode`.

    RBLN_DEVICES=8 python examples/cogvideo/verify_vae_encoder.py \
        --compiled-dir /home/seinpark/nas_data/cogvideox/compiled/vae5bi2v_f49_480x720_enc

The VAE is compiled with `uses_encoder=True` for 49 frames at 480x720, so decoder_0, decoder_n and the
encoder load together on one device. The image is a fixed PNG; the latent distribution mean (and the full
moments) are compared with the CPU result.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from cog_utils import OUT_ROOT, DeviceMemSampler, compare, host_peak_rss_gb, new_out_dir, write_meta
from diffusers import AutoencoderKLCogVideoX
from PIL import Image

from optimum.rbln import RBLNAutoencoderKLCogVideoX


MODEL_ID = "zai-org/CogVideoX-5b-I2V"
DEFAULT_IMAGE = str(OUT_ROOT / "20261013_0033_e2e_cpu_golden/frames_cpu/frame_0000.png")


def print_alloc(compiled_dir: Path) -> dict:
    import rebel

    allocs = {}
    for path in sorted(compiled_dir.glob("*.rbln")):
        alloc = rebel.RBLNCompiledModel(str(path)).get_alloc_per_node_by_key()
        gib = {key: round(sum(vals) / 2**30, 3) for key, vals in alloc.items() if sum(vals)}
        allocs[path.name] = gib
        print(f"device alloc {path.name}: total={sum(gib.values()):.3f} GiB {gib}")
    return allocs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--compiled-dir", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB").resize((args.width, args.height))
    x = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 127.5 - 1.0
    x = x[None, :, None]  # (1, 3, 1, H, W), in [-1, 1] like the pipeline's video_processor
    print(f"image={args.image} x={tuple(x.shape)} range=[{x.min():.3f}, {x.max():.3f}]")

    vae_cpu = AutoencoderKLCogVideoX.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32).eval()
    t0 = time.time()
    with torch.no_grad():
        dist_cpu = vae_cpu.encode(x).latent_dist
    print(f"CPU encode {time.time() - t0:.2f}s mean={tuple(dist_cpu.mean.shape)}")

    compiled_dir = Path(args.compiled_dir)
    compile_time = None
    with DeviceMemSampler() as mem:
        if not (compiled_dir / "rbln_config.json").exists():
            t0 = time.time()
            compiled = RBLNAutoencoderKLCogVideoX.from_pretrained(
                MODEL_ID,
                subfolder="vae",
                export=True,
                rbln_config={
                    "num_frames": args.num_frames,
                    "height": args.height,
                    "width": args.width,
                    "uses_encoder": True,
                    "create_runtimes": False,
                },
            )
            compile_time = time.time() - t0
            print(f"compiled VAE (encoder + decoder) in {compile_time:.1f}s")
            compiled.save_pretrained(compiled_dir)
            del compiled
        allocs = print_alloc(compiled_dir)
        vae_rbln = RBLNAutoencoderKLCogVideoX.from_pretrained(
            compiled_dir, export=False, rbln_config={"device": args.device}
        )
        print(f"loaded {[cfg.compiled_model_name for cfg in vae_rbln.rbln_config.compile_cfgs]} from {compiled_dir}")
        t0 = time.time()
        with torch.no_grad():
            dist_rbln = vae_rbln.encode(x).latent_dist
        rbln_time = time.time() - t0
    print(f"RBLN encode {rbln_time:.3f}s")
    print(f"device peak memory (GiB, rbln-stat): {mem.summary()}")
    print(f"RBLN dtype: {vae_rbln.rbln_config.dtype} (compiled graph), reference: CPU float32")
    mean_stats = compare("latent mean", dist_rbln.mean, dist_cpu.mean)
    moments_stats = compare("moments (mean+logvar)", dist_rbln.parameters, dist_cpu.parameters)

    out_dir = new_out_dir("vae5bi2v_encoder")
    image.save(out_dir / "input_image.png")
    torch.save(
        {"x": x, "moments_cpu": dist_cpu.parameters, "moments_rbln": dist_rbln.parameters}, out_dir / "encode.pt"
    )
    write_meta(
        out_dir,
        {
            "script": "examples/cogvideo/verify_vae_encoder.py",
            "model_id": MODEL_ID,
            "image": args.image,
            "height": args.height,
            "width": args.width,
            "compiled_num_frames": args.num_frames,
            "dtype_rbln": str(vae_rbln.rbln_config.dtype),
            "dtype_reference": "torch.float32 (CPU)",
            "compiled_dir": str(compiled_dir),
            "compile_time_s": compile_time,
            "rbln_encode_time_s": rbln_time,
            "device_alloc_gib": allocs,
            "device_peak_mem_gib": mem.summary(),
            "host_peak_rss_gb": host_peak_rss_gb(),
            "latent_mean": mean_stats,
            "moments": moments_stats,
            "notes": args.notes,
        },
    )
    print(f"saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
