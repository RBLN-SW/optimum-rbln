import argparse
import os

from transformers import AutoTokenizer

from optimum.rbln import RBLNXLMRobertaForSequenceClassification


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id BAAI/bge-reranker-v2-m3 -o bge-reranker-v2-m3 \
#       --max_seq_len 8192 --batch_size 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.model_id):
        model = RBLNXLMRobertaForSequenceClassification.from_pretrained(args.model_id)
    else:
        model = RBLNXLMRobertaForSequenceClassification.from_pretrained(
            args.model_id,
            rbln_max_seq_len=args.max_seq_len,
            rbln_batch_size=args.batch_size,
        )

    pairs = [
        [
            "what is panda?",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        ]
    ] * args.batch_size
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    inputs = tokenizer(pairs, padding="max_length", return_tensors="pt", max_length=args.max_seq_len)
    score = model(**inputs).logits.view(-1).float()
    print(score)


if __name__ == "__main__":
    main()
