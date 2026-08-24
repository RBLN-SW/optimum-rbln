import argparse
import os
import urllib

import numpy as np
import torch
from PIL import Image
from transformers import DPTImageProcessor

from optimum.rbln import RBLNDPTForDepthEstimation


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id Intel/dpt-large -o dpt-large


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Intel/dpt-large")
    return parser.parse_args()


def main():
    args = parse_args()

    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg"
    img_path = "./tabby.jpg"
    if not os.path.exists(img_path):
        with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
            f.write(response.read())
    image = Image.open(img_path)

    processor = DPTImageProcessor.from_pretrained(args.model_id)

    # A HuggingFace model id is compiled on the first run; a directory of compiled artifacts is loaded.
    model = RBLNDPTForDepthEstimation.from_pretrained(args.model_id)

    inputs = processor(images=image, return_tensors="pt")

    if model.config.is_hybrid:
        predicted_depth = model(**inputs).predicted_depth[0]
    else:
        predicted_depth = model(**inputs).predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth.save(f"depth_{os.path.basename(args.model_id)}.png")


if __name__ == "__main__":
    main()
