import argparse
import os

import torch
from huggingface_hub import hf_hub_download

from optimum.rbln import RBLNTimeSeriesTransformerForPrediction


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id huggingface/time-series-transformer-tourism-monthly \
#       -o time-series-transformer-tourism-monthly \
#       --batch_size 1 --hf-num_parallel_samples 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="huggingface/time-series-transformer-tourism-monthly")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-parallel-samples", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.model_id):
        model = RBLNTimeSeriesTransformerForPrediction.from_pretrained(args.model_id)
    else:
        model = RBLNTimeSeriesTransformerForPrediction.from_pretrained(
            args.model_id,
            rbln_batch_size=args.batch_size,
            num_parallel_samples=args.num_parallel_samples,
        )

    dataset = hf_hub_download(
        repo_id="hf-internal-testing/tourism-monthly-batch", filename="val-batch.pt", repo_type="dataset"
    )
    data = torch.load(dataset, weights_only=True)

    batched_data = {}
    for k, v in data.items():
        batched_data[k] = v[: args.batch_size]

    rbln_outputs = model.generate(**batched_data)
    mean_prediction = rbln_outputs.sequences.mean(dim=1)

    print(mean_prediction)


if __name__ == "__main__":
    main()
