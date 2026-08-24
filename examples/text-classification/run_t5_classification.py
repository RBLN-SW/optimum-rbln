import argparse
import os

from transformers import AutoTokenizer, T5EncoderModel

from optimum.rbln import RBLNT5EncoderModel


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id google-t5/t5-small -o t5-small \
#       --batch_size 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google-t5/t5-small")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt", default="Studies have been shown that owning a dog is good for you")
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if os.path.isdir(args.model_id):
        model = RBLNT5EncoderModel.from_pretrained(args.model_id)
    else:
        t5_encoder_model = T5EncoderModel.from_pretrained(args.model_id)
        model = RBLNT5EncoderModel.from_model(
            model=t5_encoder_model,
            rbln_batch_size=args.batch_size,
        )

    target_sentences = [args.prompt] * args.batch_size
    inputs = tokenizer(target_sentences, return_tensors="pt", padding="max_length", max_length=512)

    outputs = model(**inputs)
    print(outputs)


if __name__ == "__main__":
    main()
