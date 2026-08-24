import argparse
import os

from transformers import AutoTokenizer

from optimum.rbln import RBLNGptOssForCausalLM


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id openai/gpt-oss-20b -o gpt-oss-20b \
#       --max_seq_len 4096 --batch_size 1 --num_devices 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="openai/gpt-oss-20b")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-devices", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.model_id):
        # A directory of compiled artifacts (e.g. produced by optimum-rbln-cli) is loaded directly.
        model = RBLNGptOssForCausalLM.from_pretrained(args.model_id)
    else:
        # A HuggingFace model id is compiled on the first run; no separate compilation step is needed.
        model = RBLNGptOssForCausalLM.from_pretrained(
            args.model_id,
            rbln_batch_size=args.batch_size,
            rbln_max_seq_len=args.max_seq_len,
            rbln_num_devices=args.num_devices,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")

    conversations = [
        [{"role": "user", "content": "Name the largest country in the world."}],
        [{"role": "user", "content": "What is artificial intelligence?"}],
    ]
    conversations = [conversations[i % len(conversations)] for i in range(args.batch_size)]
    texts = [
        tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        for conversation in conversations
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)

    output_sequences = model.generate(**inputs, do_sample=False, max_new_tokens=128)

    for i, output in enumerate(output_sequences):
        generated = tokenizer.decode(output, skip_special_tokens=True)
        print(f"\033[32mbatch {i}:\033[0m\n{generated}\n")


if __name__ == "__main__":
    main()
