import argparse
import os

from transformers import AutoTokenizer

from optimum.rbln import RBLNBertForQuestionAnswering


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id deepset/bert-base-cased-squad2 -o bert-base-cased-squad2 \
#       --max_seq_len 512 --batch_size 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="deepset/bert-base-cased-squad2")  # or "deepset/roberta-base-squad2"
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.model_id):
        model = RBLNBertForQuestionAnswering.from_pretrained(args.model_id)
    else:
        model = RBLNBertForQuestionAnswering.from_pretrained(
            args.model_id,
            rbln_max_seq_len=512,
            rbln_batch_size=args.batch_size,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    questions = ["What is Rebellions?"] * args.batch_size
    contexts = ["Rebellions is the best NPU company."] * args.batch_size
    inputs = tokenizer(
        questions,
        contexts,
        padding="max_length",
        max_length=model.rbln_config.max_seq_len,
        truncation=True,
        return_tensors="pt",
    )

    outputs = model(**inputs)
    start_indices = outputs.start_logits.argmax(dim=-1)
    end_indices = outputs.end_logits.argmax(dim=-1)

    for i in range(args.batch_size):
        answer_ids = inputs["input_ids"][i][start_indices[i] : end_indices[i] + 1]
        print(tokenizer.decode(answer_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
