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


def test_text_encoder(
    model_id: str,
    max_seq_len: int,
    batch: int,
    threshold: float = PASS_THRESHOLD,
) -> bool:
    print("\n" + "=" * 60)
    print("Text Encoder (Qwen2_5_VLForConditionalGeneration) — golden comparison")
    print("=" * 60)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from optimum.rbln import RBLNQwen2_5_VLModel

    print("\n[1/4] Loading PyTorch Text Encoder...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.float32
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
        report("Text Encoder (last hidden state)", r, threshold)

    del text_encoder, rbln_model
    return r >= threshold


def parse_args():
    p = argparse.ArgumentParser(description="QwenImageEdit Text Encoder golden comparison (pearsonr)")
    p.add_argument("--model-id", default=MODEL_DIR, help="HuggingFace model ID or local path")
    p.add_argument("--max-seq-len", type=int, default=4096, help="Max seq len for text encoder")
    p.add_argument("--batch", type=int, default=1, help="Batch size")
    p.add_argument("--threshold", type=float, default=0.99, help="Pearson r pass threshold")
    return p.parse_args()


def main():
    args = parse_args()
    passed = test_text_encoder(
        args.model_id,
        args.max_seq_len,
        args.batch,
        threshold=args.threshold,
    )
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'text_encoder':20s}: {'PASS' if passed else 'FAIL'}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
