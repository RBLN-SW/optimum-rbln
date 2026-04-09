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
import sys

from test_qwenimage_edit_text_encoder import test_text_encoder
from test_qwenimage_edit_transformer import test_transformer
from test_qwenimage_edit_vae import test_vae

MODEL_DIR = "/mnt/shared_data/.cache/pretrained_models/NC/VARCO_I2I_inference/models"


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

    results = {}

    if args.module in ("all", "vae"):
        results["vae"] = test_vae(
            args.model_id,
            args.height,
            args.width,
            args.batch,
            threshold=args.threshold,
        )

    if args.module in ("all", "transformer"):
        results["transformer"] = test_transformer(
            args.model_id,
            args.height,
            args.width,
            args.batch,
            args.prompt_embed_length,
            threshold=args.threshold,
        )

    if args.module in ("all", "text_encoder"):
        results["text_encoder"] = test_text_encoder(
            args.model_id,
            args.max_seq_len,
            args.batch,
            threshold=args.threshold,
        )

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
