import argparse
import copy
import sys
import tempfile
import os
import shutil
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


def test_transformer(
    model_id: str,
    height: int,
    width: int,
    batch: int,
    prompt_embed_length: int,
    num_layers: int = 60,
    native: bool = False,
    compile: bool = False,
    threshold: float = PASS_THRESHOLD,
    tensor_parallel_size: int = 1,
) -> bool:
    print("\n" + "=" * 60)
    print("Transformer (QwenImageTransformer2DModel) — golden comparison")
    print("=" * 60)

    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
    from optimum.rbln import RBLNQwenImageTransformer2DModel
    vae_scale_factor = 8

    inputs = torch.load("/mnt/shared_data/groups/sw_dev/thkim/transformer_golden_inputs.pt")
    dummy_hs = inputs["hidden_states"].to(torch.bfloat16)
    dummy_enc_raw = inputs["encoder_hidden_states"].to(torch.bfloat16)
    dummy_t = inputs["timestep"].to(torch.bfloat16)

    print("\n[1/4] Loading PyTorch Transformer...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        num_layers=num_layers,
        torch_dtype=torch.bfloat16,
    )
    transformer.eval()

    lat_h = 2 * (height // (vae_scale_factor * 2))
    lat_w = 2 * (width // (vae_scale_factor * 2))
    packed_h, packed_w = lat_h // 2, lat_w // 2

    img_shapes = [[(1, packed_h, packed_w), (1, packed_h, packed_w)]] * batch
    if native:
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
        torch.save(golden_output, "golden_output.pt")
        golden_output = golden_output[0]

    print("[3/4] Compiling RBLN Transformer...")
    model_save_dir = "qwenimage_transformer_rbln"

    if compile:
        rbln_transformer = RBLNQwenImageTransformer2DModel.from_model(
            model=transformer,
            model_save_dir=model_save_dir,
            rbln_config={
                "batch_size": batch,
                "sample_size": (lat_h, lat_w),
                "prompt_embed_length": prompt_embed_length,
                "num_img_groups": 2,
                "tensor_parallel_size": tensor_parallel_size,
            },
        )
    else:
        rbln_transformer = RBLNQwenImageTransformer2DModel.from_pretrained(
            model_id=model_save_dir,
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
    if native:
        r = compute_pearsonr(golden_output.detach(), rbln_output.detach())
        print(f"Golden output: {golden_output}")
        print(f"RBLN output: {rbln_output}")  
        report("Transformer", r, threshold)

        return r >= threshold


def parse_args():
    p = argparse.ArgumentParser(description="QwenImageEdit Transformer golden comparison (pearsonr)")
    p.add_argument("--model-id", default=MODEL_DIR, help="HuggingFace model ID or local path")
    p.add_argument("--height", type=int, default=1024, help="Image height")
    p.add_argument("--width", type=int, default=1024, help="Image width")
    p.add_argument("--batch", type=int, default=1, help="Batch size")
    p.add_argument(
        "--prompt-embed-length", type=int, default=512, help="Prompt embed length for Transformer"
    )
    p.add_argument("--threshold", type=float, default=0.99, help="Pearson r pass threshold")
    p.add_argument("--num-layers", type=int, default=60, help="Number of layers to test")
    p.add_argument("--compile", action="store_true", default=False, help="Compile the transformer")
    p.add_argument("--native", action="store_true", default=False, help="Use native transformer")
    p.add_argument("--tensor-parallel-size","-tp", type=int, default=1, help="Tensor parallel size")
    return p.parse_args()


def main():
    args = parse_args()
    passed = test_transformer(
        model_id=args.model_id,
        height=args.height,
        width=args.width,
        batch=args.batch,
        prompt_embed_length=args.prompt_embed_length,
        num_layers=args.num_layers,
        native=args.native,
        compile=args.compile,
        threshold=args.threshold,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'transformer':20s}: {'PASS' if passed else 'FAIL'}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
