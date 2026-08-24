import argparse
import os

from transformers import AutoProcessor
from transformers.image_utils import load_image

from optimum.rbln import RBLNQwen3VLForConditionalGeneration


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id Qwen/Qwen3-VL-2B-Instruct -o Qwen3-VL-2B-Instruct \
#       --visual.max_seq_len 1024 --max_seq_len 16384 \
#       --kvcache_partition_len 8192 --num_devices 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument(
        "--image-url",
        default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    )
    parser.add_argument("--prompt", default="Describe this image in detail.")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.model_id):
        # A directory of compiled artifacts (e.g. produced by optimum-rbln-cli) is loaded directly.
        model = RBLNQwen3VLForConditionalGeneration.from_pretrained(args.model_id)
    else:
        # A HuggingFace model id is compiled on the first run; no separate compilation step is needed.
        model = RBLNQwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            rbln_config={
                "visual": {"max_seq_len": 1024},
                "max_seq_len": args.max_seq_len,
                "kvcache_partition_len": 8192,
                "num_devices": args.num_devices,
            },
        )

    # Cap the number of image patches so the vision tower stays within its compiled max_seq_len.
    processor = AutoProcessor.from_pretrained(args.model_id, max_pixels=768 * 256)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": load_image(args.image_url)},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    generated_ids = model.generate(**inputs, max_new_tokens=200)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    print(generated_texts[0])


if __name__ == "__main__":
    main()
