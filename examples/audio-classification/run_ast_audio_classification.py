import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoFeatureExtractor

from optimum.rbln import RBLNASTForAudioClassification


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id MIT/ast-finetuned-audioset-10-10-0.4593 \
#       -o ast-finetuned-audioset-10-10-0.4593 \
#       --batch_size 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.model_id):
        model = RBLNASTForAudioClassification.from_pretrained(args.model_id)
    else:
        model = RBLNASTForAudioClassification.from_pretrained(
            args.model_id,
            rbln_batch_size=args.batch_size,
        )

    ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    input_values_list = []
    for i in range(args.batch_size):
        input_values = feature_extractor(ds[i]["audio"]["array"], return_tensors="pt").input_values
        input_values_list.append(input_values)
    input_values = torch.cat(input_values_list, dim=0)

    rbln_logits = model(input_values)
    rbln_labels = []
    for i in range(args.batch_size):
        rbln_logit = rbln_logits[i]
        rbln_class_ids = torch.argmax(rbln_logit, dim=-1).item()
        rbln_label = model.config.id2label[rbln_class_ids]
        rbln_labels.append(rbln_label)
    print(f"labels : {rbln_labels}")


if __name__ == "__main__":
    main()
