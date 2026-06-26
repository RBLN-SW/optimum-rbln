import os
import typing

import fire
from datasets import load_dataset
from transformers import AutoProcessor

from optimum.rbln import RBLNIdefics3ForConditionalGeneration


def main(
    model_id: str = "HuggingFaceM4/Idefics3-8B-Llama3",
    batch_size: int = 1,
    from_transformers: bool = False,
    prompt: typing.Optional[str] = None,
    max_seq_len: typing.Optional[int] = None,
    tensor_parallel_size: typing.Optional[int] = 4,
):
    processor = AutoProcessor.from_pretrained(model_id)

    if from_transformers:
        model = RBLNIdefics3ForConditionalGeneration.from_pretrained(
            model_id,
            export=True,
            rbln_config={
                "text_model": {
                    "attn_impl": "flash_attn",
                    "max_seq_len": max_seq_len,
                    "use_inputs_embeds": True,
                    "tensor_parallel_size": tensor_parallel_size,
                    "batch_size": batch_size,
                }
            },
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNIdefics3ForConditionalGeneration.from_pretrained(
            os.path.basename(model_id),
            export=False,
        )

    ds = load_dataset("HuggingFaceM4/the_cauldron", "ai2d", split="train")
    samples = ds.select(range(batch_size))
    images = []
    prompts = []

    for sample in samples:
        img = sample["images"]
        images.append(img)

        message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
        prompt = processor.apply_chat_template(message, add_generation_prompt=True)
        prompts.append(prompt)

    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = dict(inputs)
    # Generate

    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    for i, text in enumerate(generated_texts):
        print(f"Sample {i + 1} generate:\n{text}\n")


if __name__ == "__main__":
    fire.Fire(main)
