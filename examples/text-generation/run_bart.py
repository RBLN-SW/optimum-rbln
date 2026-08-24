import argparse
import os

from transformers import BartTokenizer

from optimum.rbln import RBLNBartForConditionalGeneration


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id lucadiliello/bart-small -o bart-small \
#       --batch_size 1


sentences = ["UN Chief Says There Is No <mask> in Syria"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="lucadiliello/bart-small")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.model_id):
        model = RBLNBartForConditionalGeneration.from_pretrained(args.model_id)
    else:
        model = RBLNBartForConditionalGeneration.from_pretrained(
            args.model_id,
            rbln_batch_size=args.batch_size,
        )

    # Prepare inputs
    target_sentences = sentences * args.batch_size
    tokenizer = BartTokenizer.from_pretrained(args.model_id)
    inputs = tokenizer(target_sentences, return_tensors="pt", padding=True)

    # Generate
    output_sequence = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        num_beams=1,
    )

    # Decode and print the model's responses
    for i, sentence in enumerate(target_sentences):
        print("\033[94m" + sentence + " : \033[0m\n" + tokenizer.decode(output_sequence.numpy().tolist()[i]))


if __name__ == "__main__":
    main()
