import os
import typing

import fire
from datasets import load_dataset
from transformers import LlavaNextProcessor

from optimum.rbln import RBLNLlavaNextForConditionalGeneration


def main(
    model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
    batch_size: int = 1,
    from_transformers: bool = False,
    prompt: typing.Optional[str] = None,
    max_seq_len: typing.Optional[int] = None,
    tensor_parallel_size: typing.Optional[int] = 4,
    num_text_only: typing.Optional[int] = None,
):
    if from_transformers:
        processor = LlavaNextProcessor.from_pretrained(model_id)

        model = RBLNLlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            export=True,
            rbln_config={
                "language_model": {
                    "tensor_parallel_size": tensor_parallel_size,
                    "max_seq_len": max_seq_len,
                    "use_inputs_embeds": True,
                    "batch_size": batch_size,
                },
            },
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        processor = LlavaNextProcessor.from_pretrained(os.path.basename(model_id))
        model = RBLNLlavaNextForConditionalGeneration.from_pretrained(
            os.path.basename(model_id),
            export=False,
        )

    if num_text_only is None:
        num_text_only = 1 if batch_size > 1 else 0

    datasets = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train")

    conversations = []
    images = []
    for i in range(batch_size):
        if i < num_text_only:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Name the {i + 1} largest countries in the world. Explain detail.",
                        },
                    ],
                },
            ]
        else:
            idx = i * 4
            prompt = datasets[idx].get("question")
            caption = datasets[idx].get("caption")
            image = datasets[idx].get("image")
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s shown in this image?"},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f'This image shows "{caption}"'},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            images.append(image)
        conversations.append(conversation)

    prompt = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=images, text=prompt, return_tensors="pt", padding=True)

    output = model.generate(**inputs, max_new_tokens=200)

    for i in range(batch_size):
        prompt_len = inputs.input_ids[i].shape[-1]
        if i >= num_text_only:
            images[i - num_text_only].save(f"batch_{i}.png", "png")
        print(f"batch {i} -- Prompt --")
        print(processor.decode(output[i][:prompt_len], skip_special_tokens=True))
        print("---- Answer ----")
        print(processor.decode(output[i][prompt_len:], skip_special_tokens=True))


if __name__ == "__main__":
    fire.Fire(main)
