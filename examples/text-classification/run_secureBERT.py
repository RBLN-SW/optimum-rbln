import argparse
import os

import torch
from transformers import RobertaTokenizerFast

from optimum.rbln import RBLNRobertaForMaskedLM


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id ehsanaghaei/SecureBERT -o SecureBERT \
#       --max_seq_len 512 --batch_size 1


# Function to predict the masked words in a sentence
def predict(sent, tokenizer, runtime, max_seq_len=512, topk=10, print_results=True):
    inputs = tokenizer(sent, max_length=max_seq_len, padding="max_length", truncation=True, return_tensors="pt")

    masked_position = (inputs.input_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position]
    words = []

    with torch.no_grad():
        output = runtime(inputs.input_ids, inputs.attention_mask)

    last_hidden_state = output[0].squeeze()

    list_of_list = []
    for _, mask_index in enumerate(masked_pos):
        mask_hidden_state = last_hidden_state[mask_index]
        idx = torch.topk(mask_hidden_state, k=topk, dim=0)[1]
        words = [tokenizer.decode(i.item()).strip() for i in idx]
        words = [w.replace(" ", "") for w in words]
        list_of_list.append(words)
        if print_results:
            print("logits: ", mask_hidden_state[idx])
            print("predictions: ", words)

    best_guess = ""
    for j in list_of_list:
        best_guess = best_guess + "," + j[0]

    return words


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="ehsanaghaei/SecureBERT")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.model_id):
        model = RBLNRobertaForMaskedLM.from_pretrained(args.model_id)
    else:
        model = RBLNRobertaForMaskedLM.from_pretrained(
            args.model_id,
            rbln_max_seq_len=args.max_seq_len,
            rbln_batch_size=args.batch_size,
        )

    # Example sentence
    sent = []
    sent.append(
        "Gathering this information may reveal opportunities for other forms of <mask>, establishing operational resources, or initial access."
    )
    # sent.append("Information about identities may include a variety of details, including personal data as well as <mask> details such as credentials")
    # sent.append("Adversaries may also compromise sites then include <mask> content designed to collect website authentication cookies from visitors.")
    # sent.append("Adversaries may also compromise sites then include malicious content designed to collect website authentication <mask> from visitors.")
    print("SecureBERT: ")

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_id)

    # Predict masked tokens
    predict(sent, tokenizer, model)


if __name__ == "__main__":
    main()
