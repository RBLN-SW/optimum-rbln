import os
import typing

import fire
import requests
from PIL import Image
from transformers import Exaone4_5_Processor

from optimum.rbln import RBLNExaone4_5_ForConditionalGeneration


def main(
    model_id: str = "/mnt/shared_data/groups/sw_dev/.cache/pretrained_models/LGAI-EXAONE/exaone-4.5/model_32B-dummy_128k_2026-02-26",
    batch_size: int = 1,
    from_transformers: bool = False,
    max_seq_len: typing.Optional[int] = 128000,
    tensor_parallel_size: typing.Optional[int] = 16,
):
    processor = Exaone4_5_Processor.from_pretrained(model_id)

    if from_transformers:
        model = RBLNExaone4_5_ForConditionalGeneration.from_pretrained(
            model_id,
            export=True,
            rbln_config={
                "visual": {
                    "max_seq_lens": 16384, # exaone longest_edge set 28 * 28 * 4096 in preprocessor_config.json
                    "tensor_parallel_size": 4,
                },
                "kvcache_partition_len": 5120,
                "max_seq_len": max_seq_len,
                "tensor_parallel_size": tensor_parallel_size,
                "batch_size": batch_size,
            },
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNExaone4_5_ForConditionalGeneration.from_pretrained(
            os.path.basename(model_id),
            export=False,
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text] * batch_size, images=[image] * batch_size, return_tensors="pt")

    generate_ids = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    generated_texts = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    for i, text in enumerate(generated_texts):
        print(f"Sample {i + 1} generate:\n{text}\n")


if __name__ == "__main__":
    fire.Fire(main)
