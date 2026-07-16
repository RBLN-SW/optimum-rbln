import argparse
import csv
import os
import urllib.request

import numpy as np
import torch
from transformers import AutoTokenizer

from optimum.rbln import RBLNRobertaForSequenceClassification


NUM_CLASS = 4


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id cardiffnlp/twitter-roberta-base-emotion -o twitter-roberta-base-emotion \
#       --max_seq_len 512 --batch_size 1


# Preprocess text
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def download_label_mapping(task):
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode("utf-8").split("\n")
        csvreader = csv.reader(html, delimiter="\t")
    return [row[1] for row in csvreader if len(row) > 1]


def predict(text, tokenizer, model, max_seq_len, labels):
    # Encode the text
    text[0] = preprocess(text[0])
    inputs = tokenizer(
        text, max_length=max_seq_len, truncation=True, padding="max_length", return_tensors="pt"
    )

    # Run the model
    output = model(inputs.input_ids, inputs.attention_mask).logits

    for batch_itr in range(output.shape[0]):
        # Apply softmax to get probabilities
        scores = output[batch_itr].detach()
        scores = torch.nn.functional.softmax(scores, dim=-1).numpy()

        # Get ranking of scores
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        # Print out the results
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            print(f"{batch_itr}) {l} {np.round(float(s), NUM_CLASS)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="cardiffnlp/twitter-roberta-base-emotion")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    task = "emotion"

    if os.path.isdir(args.model_id):
        model = RBLNRobertaForSequenceClassification.from_pretrained(args.model_id)
    else:
        model = RBLNRobertaForSequenceClassification.from_pretrained(
            args.model_id,
            rbln_max_seq_len=args.max_seq_len,
            rbln_batch_size=args.batch_size,
        )

    prompt = ["Celebrating my promotion 😎"]

    target_sentences = prompt

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    labels = download_label_mapping(task)
    predict(target_sentences, tokenizer, model, args.max_seq_len, labels)


if __name__ == "__main__":
    main()
