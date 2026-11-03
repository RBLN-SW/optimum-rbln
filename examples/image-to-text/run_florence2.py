import argparse
import os

import torch
from PIL import Image
from transformers import AutoProcessor

from optimum.rbln import RBLNFlorence2ForConditionalGeneration


# Compile from a native HF checkpoint, or pass a saved RBLN directory to --model-id.
# python run_florence2.py --image car.jpg
# python run_florence2.py --image car.jpg --model-id Florence-2-base-rbln


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="florence-community/Florence-2-base")
    parser.add_argument("--image", required=True)
    parser.add_argument("--task", default="<CAPTION>")
    parser.add_argument("--enc-max-seq-len", type=int, default=640)
    parser.add_argument("--dec-max-seq-len", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_id, backend="pil")
    if os.path.isdir(args.model_id):
        model = RBLNFlorence2ForConditionalGeneration.from_pretrained(args.model_id)
    else:
        model = RBLNFlorence2ForConditionalGeneration.from_pretrained(
            args.model_id,
            rbln_config={
                "enc_max_seq_len": args.enc_max_seq_len,
                "dec_max_seq_len": args.dec_max_seq_len,
            },
        )
        model.save_pretrained(os.path.basename(args.model_id) + "-rbln")

    image = Image.open(args.image).convert("RGB")
    inputs = processor(text=args.task, images=image, return_tensors="pt")
    with torch.inference_mode():
        tokens = model.generate(**inputs, num_beams=1, do_sample=False, max_new_tokens=args.max_new_tokens)
    decoded = processor.batch_decode(tokens, skip_special_tokens=False)[0]
    print(processor.post_process_generation(decoded, task=args.task, image_size=image.size))


if __name__ == "__main__":
    main()
