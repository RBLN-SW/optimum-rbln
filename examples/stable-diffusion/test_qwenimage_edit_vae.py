import argparse
import sys
import tempfile

import torch
from scipy.stats import pearsonr

PASS_THRESHOLD = 0.99
MODEL_DIR = "/mnt/shared_data/.cache/pretrained_models/NC/VARCO_I2I_inference/models"


def compute_pearsonr(golden: torch.Tensor, test: torch.Tensor) -> float:
    g = golden.float().flatten().cpu().numpy()
    t = test.float().flatten().cpu().numpy()
    r, _ = pearsonr(g, t)
    return float(r)


def report(name: str, r: float, threshold: float) -> None:
    status = "PASS" if r >= threshold else "FAIL"
    print(f"  [{status}] {name}: pearsonr = {r:.6f}  (threshold = {threshold})")


def test_vae(
    model_id: str,
    height: int,
    width: int,
    batch: int,
    threshold: float = PASS_THRESHOLD,
) -> bool:
    print("\n" + "=" * 60)
    print("VAE (AutoencoderKLQwenImage) — golden comparison")
    print("=" * 60)

    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
    from optimum.rbln import RBLNAutoencoderKLQwenImage

    vae_scale_factor = 8
    z_dim = 16

    print("\n[1/4] Loading PyTorch VAE...")
    vae = AutoencoderKLQwenImage.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.eval()

    print("[2/4] Running PyTorch golden forward...")
    dummy_enc_input = torch.randn(batch, 3, 1, height, width, dtype=torch.bfloat16)
    with torch.no_grad():
        golden_enc_output = vae._encode(dummy_enc_input)

    lat_h = height // vae_scale_factor
    lat_w = width // vae_scale_factor
    dummy_dec_input = torch.randn(batch, z_dim, 1, lat_h, lat_w, dtype=torch.bfloat16)
    with torch.no_grad():
        golden_dec_output = vae._decode(dummy_dec_input, return_dict=False)
    if isinstance(golden_dec_output, tuple):
        golden_dec_output = golden_dec_output[0]

    print("[3/4] Compiling RBLN VAE...")
    with tempfile.TemporaryDirectory() as tmpdir:
        rbln_vae = RBLNAutoencoderKLQwenImage.from_model(
            model=vae,
            model_save_dir=tmpdir,
            rbln_config={
                "batch_size": batch,
                "sample_size": (height, width),
                "uses_encoder": True,
            },
        )

        print("[4/4] Running RBLN inference & comparing...")
        rbln_enc_output = rbln_vae.encoder.encode(dummy_enc_input)
        rbln_enc_moments = rbln_enc_output.mean
        golden_mean = golden_enc_output[:, :z_dim]
        r_enc = compute_pearsonr(golden_mean, rbln_enc_moments)
        report("VAE Encoder (mean)", r_enc, threshold)

        rbln_dec_output = rbln_vae.decoder.decode(dummy_dec_input)
        r_dec = compute_pearsonr(golden_dec_output, rbln_dec_output)
        report("VAE Decoder", r_dec, threshold)

    del vae, rbln_vae
    return r_enc >= threshold and r_dec >= threshold


def parse_args():
    p = argparse.ArgumentParser(description="QwenImageEdit VAE golden comparison (pearsonr)")
    p.add_argument("--model-id", default=MODEL_DIR, help="HuggingFace model ID or local path")
    p.add_argument("--height", type=int, default=1024, help="Image height")
    p.add_argument("--width", type=int, default=1024, help="Image width")
    p.add_argument("--batch", type=int, default=1, help="Batch size")
    p.add_argument("--threshold", type=float, default=0.99, help="Pearson r pass threshold")
    return p.parse_args()


def main():
    args = parse_args()
    passed = test_vae(args.model_id, args.height, args.width, args.batch, threshold=args.threshold)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'vae':20s}: {'PASS' if passed else 'FAIL'}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
