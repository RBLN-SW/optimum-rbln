import argparse
import os

from datasets import load_dataset
from transformers import AutoProcessor

from optimum.rbln import RBLNIdefics3ForConditionalGeneration


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id HuggingFaceM4/Idefics3-8B-Llama3 -o Idefics3-8B-Llama3 \
#       --text_model.attn_impl flash_attn --text_model.max_seq_len 8192 \
#       --text_model.use_inputs_embeds True --text_model.num_devices 4 \
#       --text_model.batch_size 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="HuggingFaceM4/Idefics3-8B-Llama3")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--num-devices", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    processor = AutoProcessor.from_pretrained(args.model_id)

    if os.path.isdir(args.model_id):
        model = RBLNIdefics3ForConditionalGeneration.from_pretrained(args.model_id)
    else:
        model = RBLNIdefics3ForConditionalGeneration.from_pretrained(
            args.model_id,
            rbln_config={
                "text_model": {
                    "attn_impl": "flash_attn",
                    "max_seq_len": args.max_seq_len,
                    "use_inputs_embeds": True,
                    "num_devices": args.num_devices,
                    "batch_size": args.batch_size,
                }
            },
        )

    ds = load_dataset("HuggingFaceM4/the_cauldron", "ai2d", split="train")
    samples = ds.select(range(args.batch_size))
    images = []
    prompts = []

    for sample in samples:
        img = sample["images"]
        images.append(img)

        message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
        prompt = processor.apply_chat_template(message, add_generation_prompt=True)
        prompts.append(prompt)

    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = dict(inputs)
    # Generate

    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    for i, text in enumerate(generated_texts):
        print(f"Sample {i + 1} generate:\n{text}\n")


if __name__ == "__main__":
    main()
