import argparse
import os

import numpy as np
import torch
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from transformers import pipeline

from optimum.rbln import RBLNDPTForDepthEstimation, RBLNStableDiffusionControlNetImg2ImgPipeline


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --diffusion-model-id:
#
#   optimum-rbln-cli --model-id runwayml/stable-diffusion-v1-5 -o stable-diffusion-v1-5 \
#       --unet.batch_size 2 --controlnet.batch_size 2
#   optimum-rbln-cli --model-id Intel/dpt-large -o dpt-large


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet-model-id", default="lllyasviel/control_v11f1p_sd15_depth")
    parser.add_argument("--depth-estimator-model-id", default="Intel/dpt-large")
    parser.add_argument("--prompt", default="lego batman and robin")
    return parser.parse_args()


def main():
    args = parse_args()

    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_id)

    image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg"
    )

    def get_depth_map(image, depth_estimator):
        image = depth_estimator(image)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        detected_map = torch.from_numpy(image).float() / 255.0
        depth_map = detected_map.permute(2, 0, 1)
        return depth_map

    img_width, img_height = image.size

    if os.path.isdir(args.diffusion_model_id):
        pipe = RBLNStableDiffusionControlNetImg2ImgPipeline.from_pretrained(args.diffusion_model_id)
    else:
        pipe = RBLNStableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            args.diffusion_model_id,
            controlnet=controlnet,
            rbln_img_width=img_width,
            rbln_img_height=img_height,
            rbln_config={
                "unet": {"batch_size": 2},
                "controlnet": {"batch_size": 2},
            },
            scheduler=UniPCMultistepScheduler.from_pretrained(args.diffusion_model_id, subfolder="scheduler"),
        )

    de_model = RBLNDPTForDepthEstimation.from_pretrained(args.depth_estimator_model_id)

    depth_estimator = pipeline("depth-estimation", model=de_model, image_processor=args.depth_estimator_model_id)
    depth_map = get_depth_map(image, depth_estimator).unsqueeze(0)

    image = pipe(
        prompt=args.prompt,
        image=image,
        control_image=depth_map,
        generator=torch.manual_seed(42),
    ).images[0]

    image.save(f"{args.prompt}-img2img.png")


if __name__ == "__main__":
    main()
