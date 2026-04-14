"""
QwenImageEditPlus — full pipeline submodules, logit-level golden check.

Each compiled block (``text_encoder``, ``transformer``, ``vae``) is exercised
with the same tensors as a PyTorch reference forward; outputs are compared
with Pearson r (flattened activations), matching the style of
``test_qwenimage_edit_transformer.py``.

This does **not** run tensor-parallel on the transformer (pass ``--tensor-parallel-size 1``);
it is meant to validate numerics when TP is unsupported on the compiler.

Thin layers (fast compile), aligned with ``run_qwenimage_edit.py``:

    python test_qwenimage_edit_modules.py --module all \\
        --transformer-num-layers 3 --text-encoder-num-layers 3 --vision-depth 4

Multimodal text encoder (image + prompt), closer to the real edit path:

    python test_qwenimage_edit_modules.py --module text_encoder \\
        --image-path /path/to/input.png

Usage:
    python test_qwenimage_edit_modules.py
    python test_qwenimage_edit_modules.py --module vae --height 512 --width 512
    python test_qwenimage_edit_modules.py --module transformer --golden-inputs ./inputs.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, Optional

import torch
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Defaults (override with env / CLI)
# ---------------------------------------------------------------------------

PASS_THRESHOLD = 0.99

MODEL_DIR = "/mnt/shared_data/.cache/pretrained_models/NC/VARCO_I2I_inference/models"

DEFAULT_TRANSFORMER_GOLDEN = os.environ.get(
    "QWENIMAGE_TRANSFORMER_GOLDEN",
    "/mnt/shared_data/groups/sw_dev/thkim/transformer_golden_inputs.pt",
)


def _load_thin_submodules(
    model_id: str,
    transformer_num_layers: Optional[int],
    text_encoder_num_layers: Optional[int],
    vision_depth: Optional[int],
) -> Dict[str, Any]:
    """Load submodules with reduced layers (same logic as ``run_qwenimage_edit.py``)."""
    submodules: Dict[str, Any] = {}

    if transformer_num_layers is not None:
        from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

        print(f"  [thin] transformer num_layers={transformer_num_layers}")
        submodules["transformer"] = QwenImageTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            num_layers=transformer_num_layers,
        )

    if text_encoder_num_layers is not None or vision_depth is not None:
        from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

        config = AutoConfig.from_pretrained(model_id, subfolder="text_encoder")
        if text_encoder_num_layers is not None:
            text_config = json.loads(config.text_config.to_json_string())
            text_config["num_hidden_layers"] = text_encoder_num_layers
            text_config["layer_types"] = text_config.get("layer_types", ["dense"] * 28)[:text_encoder_num_layers]
            config.text_config = type(config.text_config)(**text_config)
            config.num_hidden_layers = text_encoder_num_layers
            print(f"  [thin] text_encoder num_hidden_layers={text_encoder_num_layers}")
        if vision_depth is not None:
            vision_config = json.loads(config.vision_config.to_json_string())
            vision_config["depth"] = vision_depth
            vision_config["fullatt_block_indexes"] = [
                i for i in vision_config.get("fullatt_block_indexes", []) if i < vision_depth
            ]
            if not vision_config["fullatt_block_indexes"]:
                vision_config["fullatt_block_indexes"] = [vision_depth - 1]
            config.vision_config = type(config.vision_config)(**vision_config)
            print(f"  [thin] text_encoder vision depth={vision_depth}")

        submodules["text_encoder"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            subfolder="text_encoder",
            config=config,
        )

    return submodules


def compute_pearsonr(golden: torch.Tensor, test: torch.Tensor) -> float:
    g = golden.float().flatten().cpu().numpy()
    t = test.float().flatten().cpu().numpy()
    r, _ = pearsonr(g, t)
    return float(r)


def report(name: str, r: float, threshold: float) -> None:
    status = "PASS" if r >= threshold else "FAIL"
    print(f"  [{status}] {name}: pearsonr = {r:.6f}  (threshold = {threshold})")


def _build_text_encoder_inputs(
    processor: Any,
    prompt: str,
    image_path: Optional[str],
) -> Dict[str, torch.Tensor]:
    """Match QwenImage-style system + user message; optional real image for VL path."""
    if image_path:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": "Describe the key features of the input image."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        batch = processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors="pt",
        )
    else:
        template = (
            "<|im_start|>system\nDescribe the key features of the input image.<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        txt = [template.format(prompt)]
        batch = processor(text=txt, padding=True, return_tensors="pt")

    return {k: v for k, v in batch.items() if v is not None and isinstance(v, torch.Tensor)}


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


def test_vae(model_id: str, height: int, width: int, batch: int, threshold: float) -> bool:
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
        report("VAE Encoder (mean)", r_enc, threshold)

        rbln_dec_output = rbln_vae.decoder.decode(dummy_dec_input)
        r_dec = compute_pearsonr(golden_dec_output, rbln_dec_output)
        report("VAE Decoder", r_dec, threshold)

    del vae, rbln_vae
    return r_enc >= threshold and r_dec >= threshold


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


def test_transformer(
    model_id: str,
    height: int,
    width: int,
    batch: int,
    prompt_embed_length: int,
    threshold: float,
    *,
    transformer_num_layers: Optional[int],
    golden_inputs_path: Optional[str],
    use_bf16: bool,
    tensor_parallel_size: int,
) -> bool:
    print("\n" + "=" * 60)
    print("Transformer (QwenImageTransformer2DModel) — golden comparison")
    print("=" * 60)

    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
    from optimum.rbln import RBLNQwenImageTransformer2DModel

    vae_scale_factor = 8
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    thin = _load_thin_submodules(
        model_id,
        transformer_num_layers,
        text_encoder_num_layers=None,
        vision_depth=None,
    )

    print("\n[1/4] Loading PyTorch Transformer...")
    if "transformer" in thin:
        transformer = thin["transformer"].to(dtype=dtype)
    else:
        kw: Dict[str, Any] = {"torch_dtype": dtype}
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer", **kw
        )
    transformer.eval()

    lat_h = 2 * (height // (vae_scale_factor * 2))
    lat_w = 2 * (width // (vae_scale_factor * 2))
    packed_h, packed_w = lat_h // 2, lat_w // 2
    total_seq_len = packed_h * packed_w * 2

    path = golden_inputs_path or DEFAULT_TRANSFORMER_GOLDEN
    if path and os.path.isfile(path):
        print(f"  Using golden inputs: {path}")
        inputs = torch.load(path, weights_only=False)
    else:
        print(
            f"  [warn] Golden file not found ({path}); using synthetic tensors "
            f"(shape-only; pearsonr vs random reference will fail unless you disable native compare)."
        )
        in_ch = transformer.config.in_channels
        inputs = {
            "hidden_states": torch.randn(batch, total_seq_len, in_ch, dtype=dtype),
            "encoder_hidden_states": torch.randn(batch, min(prompt_embed_length, 128), transformer.config.joint_attention_dim, dtype=dtype),
            "timestep": torch.ones(batch, dtype=dtype),
        }

    dummy_hs = inputs["hidden_states"].to(dtype=dtype)
    dummy_enc_raw = inputs["encoder_hidden_states"].to(dtype=dtype)
    dummy_t = inputs.get("timestep", torch.ones(batch, dtype=dtype)).to(dtype=dtype)
    if dummy_t.ndim == 0:
        dummy_t = dummy_t.expand(batch)

    img_shapes = [[(1, packed_h, packed_w), (1, packed_h, packed_w)]] * batch

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
                "tensor_parallel_size": tensor_parallel_size,
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
        report("Transformer (hidden_states)", r, threshold)

    del transformer, rbln_transformer
    return r >= threshold


# ---------------------------------------------------------------------------
# Text encoder
# ---------------------------------------------------------------------------


def test_text_encoder(
    model_id: str,
    max_seq_len: int,
    batch: int,
    threshold: float,
    *,
    text_encoder_num_layers: Optional[int],
    vision_depth: Optional[int],
    image_path: Optional[str],
    visual_max_seq_lens: int,
) -> bool:
    print("\n" + "=" * 60)
    print("Text Encoder (Qwen2_5_VL) — last hidden state golden comparison")
    print("=" * 60)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from optimum.rbln import RBLNQwen2_5_VLModel

    thin = _load_thin_submodules(
        model_id,
        transformer_num_layers=None,
        text_encoder_num_layers=text_encoder_num_layers,
        vision_depth=vision_depth,
    )

    print("\n[1/4] Loading PyTorch Text Encoder...")
    if "text_encoder" in thin:
        text_encoder = thin["text_encoder"]
    else:
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=torch.float32,
        )
    text_encoder.eval()

    processor = AutoProcessor.from_pretrained(model_id, subfolder="processor")

    print("[2/4] Preparing inputs & running PyTorch golden forward...")
    prompt = "Make the cat wear a hat"
    model_inputs = _build_text_encoder_inputs(processor, prompt, image_path)
    fwd_kw = {k: v for k, v in model_inputs.items() if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}

    with torch.no_grad():
        golden_outputs = text_encoder(**fwd_kw, output_hidden_states=True)
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
                    "max_seq_lens": visual_max_seq_lens,
                },
            },
        )

        print("[4/4] Running RBLN inference & comparing...")
        rbln_outputs = rbln_model(**fwd_kw, output_hidden_states=True)
        rbln_hidden = rbln_outputs.hidden_states[-1]

        seq_len = int(model_inputs["attention_mask"].sum().item())
        golden_valid = golden_hidden[0, :seq_len]
        rbln_valid = rbln_hidden[0, :seq_len]

        r = compute_pearsonr(golden_valid, rbln_valid)
        report("Text Encoder (last hidden state)", r, threshold)

    del text_encoder, rbln_model
    return r >= threshold


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="QwenImageEditPlus: per-submodule logit-level golden comparison (Pearson r)."
    )
    p.add_argument("--model-id", default=MODEL_DIR, help="HF model id or local checkpoint directory")
    p.add_argument(
        "--module",
        default="all",
        choices=["all", "vae", "transformer", "text_encoder"],
        help="Submodule to test",
    )
    p.add_argument("--height", type=int, default=1024, help="Image height for VAE / transformer latent math")
    p.add_argument("--width", type=int, default=1024, help="Image width for VAE / transformer latent math")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prompt-embed-length", type=int, default=512, help="Compiled prompt length for transformer")
    p.add_argument("--max-seq-len", type=int, default=4096, help="Text encoder max sequence length")
    p.add_argument(
        "--visual-max-seq-lens",
        type=int,
        default=6400,
        help="Vision branch max_seq_lens for RBLN text encoder (match run_qwenimage_edit.py)",
    )
    p.add_argument("--threshold", type=float, default=0.99, help="Pearson r pass threshold")
    p.add_argument(
        "--transformer-num-layers",
        type=int,
        default=None,
        help="If set, load transformer with this many layers (fast compile)",
    )
    p.add_argument(
        "--text-encoder-num-layers",
        type=int,
        default=None,
        help="If set, shrink language backbone to this many layers",
    )
    p.add_argument(
        "--vision-depth",
        type=int,
        default=None,
        help="If set, shrink vision tower depth (with --text-encoder-num-layers or alone)",
    )
    p.add_argument(
        "--golden-inputs",
        default=None,
        help=f"transformer_golden_inputs.pt path (default: env QWENIMAGE_TRANSFORMER_GOLDEN or {DEFAULT_TRANSFORMER_GOLDEN})",
    )
    p.add_argument(
        "--image-path",
        default=None,
        help="If set, text_encoder test uses image+prompt (multimodal); same as edit pipeline",
    )
    p.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=1,
        help="Transformer compile tensor parallel size (use 1 if compiler has no TP support)",
    )
    p.add_argument(
        "--no-bf16-transformer",
        action="store_true",
        help="Run transformer in float32 instead of bfloat16",
    )
    return p.parse_args()


def main():
    args = parse_args()
    threshold = args.threshold

    results = {}

    if args.module in ("all", "vae"):
        results["vae"] = test_vae(args.model_id, args.height, args.width, args.batch, threshold)

    if args.module in ("all", "transformer"):
        results["transformer"] = test_transformer(
            args.model_id,
            args.height,
            args.width,
            args.batch,
            args.prompt_embed_length,
            threshold,
            transformer_num_layers=args.transformer_num_layers,
            golden_inputs_path=args.golden_inputs,
            use_bf16=not args.no_bf16_transformer,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    if args.module in ("all", "text_encoder"):
        if args.text_encoder_num_layers is None and args.vision_depth is None:
            print(
                "\n[hint] Full text_encoder compile is heavy; consider "
                "`--text-encoder-num-layers 2 --vision-depth 4` like run_qwenimage_edit.py thin mode."
            )
        results["text_encoder"] = test_text_encoder(
            args.model_id,
            args.max_seq_len,
            args.batch,
            threshold,
            text_encoder_num_layers=args.text_encoder_num_layers,
            vision_depth=args.vision_depth,
            image_path=args.image_path,
            visual_max_seq_lens=args.visual_max_seq_lens,
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
        print("\nAll selected modules passed golden comparison.")
    else:
        print("\nSome modules FAILED. See logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
