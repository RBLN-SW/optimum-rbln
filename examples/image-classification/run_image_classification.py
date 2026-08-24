import argparse
import os
import urllib

from PIL import Image
from transformers import AutoImageProcessor

from optimum.rbln import RBLNResNetForImageClassification


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id microsoft/resnet-50 -o resnet-50 \
#       --image_size 224 --batch_size 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="microsoft/resnet-50")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg"
    img_path = "./tabby.jpg"
    if not os.path.exists(img_path):
        with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
            f.write(response.read())

    image = Image.open(img_path)

    if os.path.isdir(args.model_id):
        model = RBLNResNetForImageClassification.from_pretrained(args.model_id)
    else:
        model = RBLNResNetForImageClassification.from_pretrained(
            args.model_id,
            rbln_image_size=224,
            rbln_batch_size=args.batch_size,
        )

    image_processor = AutoImageProcessor.from_pretrained(args.model_id)
    inputs = image_processor([image] * args.batch_size, return_tensors="pt")

    logits = model(**inputs).logits
    labels = logits.argmax(-1)

    print("predicted label:", [model.config.id2label[label.item()] for label in labels])


if __name__ == "__main__":
    main()
