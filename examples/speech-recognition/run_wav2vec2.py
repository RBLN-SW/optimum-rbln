import argparse
import os

import torch
from datasets import load_dataset
from transformers import Wav2Vec2Processor

from optimum.rbln import RBLNWav2Vec2ForCTC


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id facebook/wav2vec2-base-960h -o wav2vec2-base-960h \
#       --batch_size 1 --max_seq_len 160005


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    processor = Wav2Vec2Processor.from_pretrained(args.model_id)

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    input_values = []
    for i in range(args.batch_size):
        input_value = processor(
            ds[i]["audio"]["array"], return_tensors="pt", padding="max_length", max_length=160005, truncation=True
        ).input_values
        input_values.append(input_value)
    input_values = torch.cat(input_values, dim=0)

    if os.path.isdir(args.model_id):
        model = RBLNWav2Vec2ForCTC.from_pretrained(args.model_id)
    else:
        model = RBLNWav2Vec2ForCTC.from_pretrained(
            args.model_id,
            rbln_batch_size=args.batch_size,
            rbln_max_seq_len=160005,
        )

    output = model(input_values)
    predicted_ids = torch.argmax(output.logits, dim=-1)
    transcriptions = processor.batch_decode(predicted_ids)
    for i, transcription in enumerate(transcriptions):
        print(f"transcription_{i} : {transcription}")


if __name__ == "__main__":
    main()
