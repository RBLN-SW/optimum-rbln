"""Compare the RBLN CogVideoX transformer with the CPU fp32 `CogVideoXTransformer3DModel`.

The reference always runs in float32 on the CPU. The RBLN graph is compiled at --torch-dtype (default: the
model card dtype, float16 for CogVideoX-2b and bfloat16 for CogVideoX-5b-I2V).

Reduced-layer compile check with random weights (no checkpoint load):
    RBLN_DEVICES=12 python examples/cogvideo/verify_transformer.py --num-layers 1 --random-init

Full model with the THUDM/CogVideoX-2b weights:
    RBLN_DEVICES=12 python examples/cogvideo/verify_transformer.py --num-layers 30 \
        --compiled-dir /home/seinpark/nas_data/cogvideox/compiled/transformer_l30

Inputs are fixed-seed random tensors (latent, encoder_hidden_states) and a fixed timestep.
Other config values keep the real CogVideoX-2b values.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from cog_utils import (
    TORCH_DTYPE_CHOICES,
    DeviceMemSampler,
    compare,
    dtype_name,
    host_peak_rss_gb,
    new_out_dir,
    resolve_torch_dtype,
    write_meta,
)
from diffusers import CogVideoXTransformer3DModel

from optimum.rbln import RBLNCogVideoXTransformer3DModel


MODEL_ID = "THUDM/CogVideoX-2b"


def rope_tables(cfg, num_latent_frames: int):
    """The RoPE (cos, sin) tables, as CogVideoXPipeline._prepare_rotary_positional_embeddings builds them (1.0)."""
    from diffusers.models.embeddings import get_3d_rotary_pos_embed
    from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid

    p = cfg.patch_size
    grid_h, grid_w = cfg.sample_height // p, cfg.sample_width // p
    crops = get_resize_crop_region_for_grid((grid_h, grid_w), grid_w, grid_h)
    return get_3d_rotary_pos_embed(
        embed_dim=cfg.attention_head_dim,
        crops_coords=crops,
        grid_size=(grid_h, grid_w),
        temporal_size=num_latent_frames,
        device=torch.device("cpu"),
    )


def print_alloc(compiled_dir: Path) -> None:
    import rebel

    for path in sorted(compiled_dir.glob("*.rbln")):
        alloc = rebel.RBLNCompiledModel(str(path)).get_alloc_per_node_by_key()
        gib = {key: [round(v / 2**30, 3) for v in vals] for key, vals in alloc.items() if sum(vals)}
        total = sum(sum(vals) for vals in alloc.values()) / 2**30
        print(f"device alloc {path.name}: total={total:.3f} GiB {gib}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--num-layers", type=int, default=None, help="default: the checkpoint's num_layers")
    parser.add_argument("--random-init", action="store_true", help="random weights from the config (no load)")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compiled-dir", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="num_devices for the compile")
    parser.add_argument(
        "--torch-dtype", choices=TORCH_DTYPE_CHOICES, default="auto", help="compile dtype (auto: model card dtype)"
    )
    parser.add_argument("--timeout", type=int, default=60, help="runtime timeout in seconds")
    parser.add_argument("--desc", type=str, default=None)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    model_id = args.model_id
    torch_dtype = resolve_torch_dtype(model_id, args.torch_dtype)
    config = CogVideoXTransformer3DModel.load_config(model_id, subfolder="transformer")
    if args.num_layers is None:
        args.num_layers = config["num_layers"]
    if args.random_init:
        torch.manual_seed(args.seed)
        model = CogVideoXTransformer3DModel.from_config({**config, "num_layers": args.num_layers})
    else:
        kwargs = {} if args.num_layers == config["num_layers"] else {"num_layers": args.num_layers}
        model = CogVideoXTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.float32, low_cpu_mem_usage=False, **kwargs
        )
    model = model.to(torch.float32).eval()
    cfg = model.config
    print(
        f"model: num_layers={cfg.num_layers} heads={cfg.num_attention_heads} head_dim={cfg.attention_head_dim} "
        f"text_embed_dim={cfg.text_embed_dim} max_text_seq_length={cfg.max_text_seq_length} "
        f"patch_size={cfg.patch_size} in_channels={cfg.in_channels} rope={cfg.use_rotary_positional_embeddings} "
        f"weights={'random' if args.random_init else model_id}"
    )

    num_latent = (args.num_frames - 1) // cfg.temporal_compression_ratio + 1
    gen = torch.Generator().manual_seed(args.seed)
    hidden_states = torch.randn(
        args.batch_size, num_latent, cfg.in_channels, cfg.sample_height, cfg.sample_width, generator=gen
    )
    encoder_hidden_states = torch.randn(args.batch_size, cfg.max_text_seq_length, cfg.text_embed_dim, generator=gen)
    timestep = torch.full((args.batch_size,), args.timestep, dtype=torch.int64)
    rope = rope_tables(cfg, num_latent) if cfg.use_rotary_positional_embeddings else None
    if rope is not None:
        print(f"image_rotary_emb cos/sin={tuple(rope[0].shape)} {rope[0].dtype}")
    print(
        f"seed={args.seed} hidden_states={tuple(hidden_states.shape)} "
        f"encoder_hidden_states={tuple(encoder_hidden_states.shape)} timestep={timestep.tolist()}"
    )

    t0 = time.time()
    with torch.no_grad():
        ref = model(hidden_states, encoder_hidden_states, timestep, image_rotary_emb=rope, return_dict=False)[0]
    cpu_time = time.time() - t0
    print(f"CPU fp32 forward {cpu_time:.1f}s out={tuple(ref.shape)}")

    compiled_dir = Path(args.compiled_dir) if args.compiled_dir else None
    rbln_config = {"batch_size": args.batch_size, "num_frames": args.num_frames, "device": args.device}
    if args.tensor_parallel_size:
        rbln_config["num_devices"] = args.tensor_parallel_size
        rbln_config["device"] = list(range(args.device, args.device + args.tensor_parallel_size))
    compile_time = None
    with DeviceMemSampler() as mem:
        if compiled_dir is None or not (compiled_dir / "rbln_config.json").exists():
            # Compile without runtimes and save first, so the artifact survives a failed device run.
            compiled_dir = compiled_dir or Path(new_out_dir("transformer_compiled_tmp"))
            t0 = time.time()
            # The CPU reference is done, so cast the model in place to the compile dtype.
            compiled = RBLNCogVideoXTransformer3DModel.from_model(
                model.to(torch_dtype),
                rbln_config={**{k: v for k, v in rbln_config.items() if k != "device"}, "create_runtimes": False},
            )
            compile_time = time.time() - t0
            print(f"compiled transformer (num_layers={cfg.num_layers}) in {compile_time:.1f}s")
            compiled.save_pretrained(compiled_dir)
            print(f"saved compiled transformer to {compiled_dir}")
            del compiled
        print_alloc(compiled_dir)
        t0 = time.time()
        rbln = RBLNCogVideoXTransformer3DModel.from_pretrained(
            compiled_dir, export=False, rbln_config={"device": rbln_config["device"], "timeout": args.timeout}
        )
        print(f"loaded compiled transformer from {compiled_dir} in {time.time() - t0:.1f}s")
        compile_rss = host_peak_rss_gb()

        t0 = time.time()
        out = rbln(hidden_states, encoder_hidden_states, timestep, image_rotary_emb=rope, return_dict=False)[0]
        rbln_time = time.time() - t0
    print(f"RBLN forward {rbln_time:.2f}s out={tuple(out.shape)} dtype={out.dtype}")
    print(f"host peak RSS: {compile_rss:.2f} GB")
    print(f"device peak memory (GiB, rbln-stat): {mem.summary()}")
    print(f"RBLN dtype: {rbln.rbln_config.dtype} (compiled graph), reference: CPU float32")
    stats = compare(f"transformer L{cfg.num_layers}", out, ref)
    per_frame = [compare(f"latent frame {f:02d}", out[:, f], ref[:, f])["pearson_r"] for f in range(out.shape[1])]
    print("per-latent-frame r: " + " ".join(f"{r:.5f}" for r in per_frame))

    weights = "rand" if args.random_init else "real"
    desc = args.desc or f"transformer_L{cfg.num_layers}_{weights}_{dtype_name(rbln.rbln_config.dtype)}"
    out_dir = new_out_dir(desc)
    torch.save(
        {"hidden_states": hidden_states, "encoder_hidden_states": encoder_hidden_states, "timestep": timestep},
        out_dir / "inputs.pt",
    )
    torch.save({"rbln": out, "cpu": ref}, out_dir / "outputs.pt")
    write_meta(
        out_dir,
        {
            "script": "examples/cogvideo/verify_transformer.py",
            "model_id": model_id,
            "weights": "random" if args.random_init else "pretrained",
            "num_layers": cfg.num_layers,
            "seed": args.seed,
            "num_frames": args.num_frames,
            "num_latent_frames": num_latent,
            "hidden_states_shape": list(hidden_states.shape),
            "encoder_hidden_states_shape": list(encoder_hidden_states.shape),
            "timestep": timestep.tolist(),
            "dtype_rbln": str(rbln.rbln_config.dtype),
            "dtype_reference": "torch.float32 (CPU)",
            "tensor_parallel_size": args.tensor_parallel_size,
            "compiled_graphs": [cfg.compiled_model_name for cfg in rbln.rbln_config.compile_cfgs],
            "compiled_dir": str(compiled_dir) if compiled_dir else None,
            "compile_time_s": compile_time,
            "rbln_forward_time_s": rbln_time,
            "cpu_forward_time_s": cpu_time,
            "host_peak_rss_gb": compile_rss,
            "device_peak_mem_gib": mem.summary(),
            "metrics": stats,
            "per_latent_frame_r": per_frame,
            "notes": args.notes,
        },
    )
    print(f"saved outputs to {out_dir}")
    print(json.dumps({"r": stats["pearson_r"], "out_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
