"""
QwenImageEditPlus — per-module compile & inference golden comparison.

Compiles each submodule (text_encoder, transformer, vae) individually via
optimum-rbln, runs inference with identical inputs, and compares RBLN output
against the PyTorch golden output using Pearson correlation.

Usage:
    python test_qwenimage_edit_modules.py
    python test_qwenimage_edit_modules.py --module vae
    python test_qwenimage_edit_modules.py --module transformer --height 128 --width 128
    python test_qwenimage_edit_modules.py --module text_encoder --max-seq-len 4096
"""

import argparse
import math
import sys
import tempfile
from typing import Tuple

import torch
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# NOTE: The RBLN transformer compilation applies three internal patches
# (real RoPE, arithmetic _modulate, pre-computed RoPE buffers) inside
# get_compiled_model(). These patches are mathematically equivalent to
# the original ops, so the unpatched PyTorch golden output should match
# the RBLN output within pearsonr tolerance.

PASS_THRESHOLD = 0.99

MODEL_DIR = "/mnt/shared_data/.cache/pretrained_models/NC/VARCO_I2I_inference/models"


def compute_pearsonr(golden: torch.Tensor, test: torch.Tensor) -> float:
    g = golden.float().flatten().cpu().numpy()
    t = test.float().flatten().cpu().numpy()
    r, _ = pearsonr(g, t)
    return float(r)


def report(name: str, r: float) -> None:
    status = "PASS" if r >= PASS_THRESHOLD else "FAIL"
    print(f"  [{status}] {name}: pearsonr = {r:.6f}  (threshold = {PASS_THRESHOLD})")


def calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int]:
    w = math.sqrt(target_area * ratio)
    h = w / ratio
    w = round(w / 32) * 32
    h = round(h / 32) * 32
    return int(w), int(h)


# ---------------------------------------------------------------------------
# VAE test
# ---------------------------------------------------------------------------

def test_vae(model_id: str, height: int, width: int, batch: int):
    print("\n" + "=" * 60)
    print("VAE (AutoencoderKLQwenImage) — golden comparison")
    print("=" * 60)

    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
    from optimum.rbln import RBLNAutoencoderKLQwenImage

    vae_scale_factor = 8
    z_dim = 16

    print("\n[1/4] Loading PyTorch VAE...")
    vae = AutoencoderKLQwenImage.from_pretrained(model_id, subfolder="vae")
    vae.eval()

    print("[2/4] Running PyTorch golden forward...")
    dummy_enc_input = torch.randn(batch, 3, 1, height, width)
    with torch.no_grad():
        golden_enc_output = vae._encode(dummy_enc_input)

    lat_h = height // vae_scale_factor
    lat_w = width // vae_scale_factor
    dummy_dec_input = torch.randn(batch, z_dim, 1, lat_h, lat_w)
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
        report("VAE Encoder (mean)", r_enc)

        rbln_dec_output = rbln_vae.decoder.decode(dummy_dec_input)
        r_dec = compute_pearsonr(golden_dec_output, rbln_dec_output)
        report("VAE Decoder", r_dec)

    del vae, rbln_vae
    return r_enc >= PASS_THRESHOLD and r_dec >= PASS_THRESHOLD


# ---------------------------------------------------------------------------
# Transformer test
# ---------------------------------------------------------------------------

def test_transformer(model_id: str, height: int, width: int, batch: int, prompt_embed_length: int):
    print("\n" + "=" * 60)
    print("Transformer (QwenImageTransformer2DModel) — golden comparison")
    print("=" * 60)

    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
    from optimum.rbln import RBLNQwenImageTransformer2DModel

    vae_scale_factor = 8

    print("\n[1/4] Loading PyTorch Transformer...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer", num_layers=1,
    )
    transformer.eval()

    lat_h = 2 * (height // (vae_scale_factor * 2))
    lat_w = 2 * (width // (vae_scale_factor * 2))
    packed_h, packed_w = lat_h // 2, lat_w // 2
    total_seq_len = packed_h * packed_w * 2

    img_shapes = [[(1, packed_h, packed_w), (1, packed_h, packed_w)]] * batch
    inputs = torch.load("/mnt/shared_data/groups/sw_dev/thkim/transformer_golden_inputs.pt")
    dummy_hs = inputs["hidden_states"]
    dummy_enc_raw = inputs["encoder_hidden_states"]
    dummy_t = torch.tensor([1.0] * batch)

    actual_enc_len = dummy_enc_raw.shape[1]
    if actual_enc_len < prompt_embed_length:
        pad_len = prompt_embed_length - actual_enc_len
        dummy_enc = torch.nn.functional.pad(dummy_enc_raw, (0, 0, 0, pad_len), value=0.0)
        enc_mask = torch.zeros(batch, prompt_embed_length, dtype=torch.bool)
        enc_mask[:, :actual_enc_len] = True
    else:
        dummy_enc = dummy_enc_raw
        enc_mask = None

    print("[2/4] Running PyTorch golden forward...")

    with torch.no_grad():
        golden_output = transformer(
            hidden_states=dummy_hs,
            encoder_hidden_states=dummy_enc,
            encoder_hidden_states_mask=enc_mask,
            timestep=dummy_t,
            img_shapes=img_shapes,
            guidance=None,
            return_dict=False,
        )
    golden_output = golden_output[0]

    print("[3/4] Compiling RBLN Transformer...")
    with tempfile.TemporaryDirectory() as tmpdir:
        rbln_transformer = RBLNQwenImageTransformer2DModel.from_model(
            model=transformer,
            model_save_dir=tmpdir,
            rbln_config={
                "batch_size": batch,
                "sample_size": (lat_h, lat_w),
                "prompt_embed_length": prompt_embed_length,
                "num_img_groups": 2,
            },
        )

        print("[4/4] Running RBLN inference & comparing...")
        rbln_output = rbln_transformer(
            hidden_states=dummy_hs,
            encoder_hidden_states=dummy_enc_raw,
            timestep=dummy_t,
            return_dict=False,
        )
        if isinstance(rbln_output, tuple):
            rbln_output = rbln_output[0]

        r = compute_pearsonr(golden_output.detach(), rbln_output.detach())
        report("Transformer", r)

    print(f"Golden output: {golden_output}")
    print(f"RBLN output: {rbln_output}")  
    del transformer, rbln_transformer
    return r >= PASS_THRESHOLD


# ---------------------------------------------------------------------------
# Text encoder test
# ---------------------------------------------------------------------------

def test_text_encoder(model_id: str, max_seq_len: int, batch: int):
    print("\n" + "=" * 60)
    print("Text Encoder (Qwen2_5_VLForConditionalGeneration) — golden comparison")
    print("=" * 60)

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from optimum.rbln import RBLNQwen2_5_VLModel

    print("\n[1/4] Loading PyTorch Text Encoder...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder",
    )
    text_encoder.eval()

    processor = AutoProcessor.from_pretrained(model_id, subfolder="processor")

    print("[2/4] Preparing inputs & running PyTorch golden forward...")
    prompt = "Make the cat wear a hat"
    template = (
        "<|im_start|>system\nDescribe the key features of the input image.<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    txt = [template.format(prompt)]

    model_inputs = processor(text=txt, padding=True, return_tensors="pt")

    with torch.no_grad():
        golden_outputs = text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            output_hidden_states=True,
        )
    golden_hidden = golden_outputs.hidden_states[-1]

    print("[3/4] Compiling RBLN Text Encoder (RBLNQwen2_5_VLModel)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        rbln_model = RBLNQwen2_5_VLModel.from_model(
            model=text_encoder,
            model_save_dir=tmpdir,
            rbln_config={
                "max_seq_len": max_seq_len,
                "batch_size": batch,
                "use_inputs_embeds": True,
                "output_hidden_states": True,
                "visual": {
                    "max_seq_lens": 6400,
                },
            },
        )

        print("[4/4] Running RBLN inference & comparing...")
        rbln_outputs = rbln_model(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            output_hidden_states=True,
        )
        rbln_hidden = rbln_outputs.hidden_states[-1]

        seq_len = model_inputs.attention_mask.sum().item()
        golden_valid = golden_hidden[0, :seq_len]
        rbln_valid = rbln_hidden[0, :seq_len]

        r = compute_pearsonr(golden_valid, rbln_valid)
        report("Text Encoder (last hidden state)", r)

    del text_encoder, rbln_model
    return r >= PASS_THRESHOLD


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="QwenImageEditPlus per-module golden comparison (pearsonr)"
    )
    p.add_argument(
        "--model-id", default=MODEL_DIR,
        help="HuggingFace model ID or local path",
    )
    p.add_argument(
        "--module", default="all", choices=["all", "vae", "transformer", "text_encoder"],
        help="Which module to test (default: all)",
    )
    p.add_argument("--height", type=int, default=1024, help="Image height for VAE/Transformer test")
    p.add_argument("--width", type=int, default=1024, help="Image width for VAE/Transformer test")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prompt-embed-length", type=int, default=512, help="Prompt embed length for Transformer")
    p.add_argument("--max-seq-len", type=int, default=4096, help="Max seq len for text encoder")
    p.add_argument("--threshold", type=float, default=0.99, help="Pearson r pass threshold")
    return p.parse_args()


def main():
    args = parse_args()
    global PASS_THRESHOLD
    PASS_THRESHOLD = args.threshold

    results = {}

    if args.module in ("all", "vae"):
        results["vae"] = test_vae(args.model_id, args.height, args.width, args.batch)

    if args.module in ("all", "transformer"):
        results["transformer"] = test_transformer(
            args.model_id, args.height, args.width, args.batch, args.prompt_embed_length
        )

    if args.module in ("all", "text_encoder"):
        results["text_encoder"] = test_text_encoder(args.model_id, args.max_seq_len, args.batch)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll modules passed golden comparison.")
    else:
        print("\nSome modules FAILED. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
