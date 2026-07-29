import argparse
import os

import torch
from transformers import AutoTokenizer

from optimum.rbln import RBLNXLMRobertaModel


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id BAAI/bge-m3 -o bge-m3 \
#       --max_seq_len 8192 --batch_size 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="BAAI/bge-m3")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.model_id):
        model = RBLNXLMRobertaModel.from_pretrained(args.model_id)
    else:
        model = RBLNXLMRobertaModel.from_pretrained(
            args.model_id,
            rbln_max_seq_len=args.max_seq_len,
            rbln_batch_size=args.batch_size,
        )

    input_q = ["what is panda?"] * args.batch_size
    input_m = [
        "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
    ] * args.batch_size
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    input_q = tokenizer(input_q, padding="max_length", return_tensors="pt", max_length=args.max_seq_len)
    input_m = tokenizer(input_m, padding="max_length", return_tensors="pt", max_length=args.max_seq_len)

    q_output = model(input_q.input_ids, input_q.attention_mask)
    m_output = model(input_m.input_ids, input_m.attention_mask)
    q_output = torch.nn.functional.normalize(q_output[0][:, 0], dim=-1)
    m_output = torch.nn.functional.normalize(m_output[0][:, 0], dim=-1)

    score = q_output @ m_output.T
    print(score)


if __name__ == "__main__":
    main()
