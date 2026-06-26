import os
import typing

import fire
from transformers import AutoTokenizer

from optimum.rbln import RBLNLlamaForCausalLM


def main(
    model_id: str = "meta-llama/Llama-2-7b-chat-hf",
    batch_size: int = 1,
    from_transformers: bool = False,
    max_seq_len: typing.Optional[int] = None,
    tensor_parallel_size: typing.Optional[int] = 4,
):
    # Example input sentences for the model
    sentences = [
        [{"role": "user", "content": "Name the largest country in the world?"}],
        [{"role": "user", "content": "What is Artificial intelligence?"}],
    ]

    if from_transformers:
        # Compile the RBLN-optimized Llama model (if export=True)
        model = RBLNLlamaForCausalLM.from_pretrained(
            model_id=model_id,
            export=True,
            # The following arguments are specific to RBLN compilation
            rbln_batch_size=batch_size,
            rbln_max_seq_len=max_seq_len,
            rbln_tensor_parallel_size=tensor_parallel_size,
            torch_dtype="auto",
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        # Load compiled model
        model = RBLNLlamaForCausalLM.from_pretrained(model_id=os.path.basename(model_id), export=False)

    # Prepare inputs
    sentences = [sentences[i % 2] for i in range(batch_size)]
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    texts = [
        tokenizer.apply_chat_template(sentence, add_generation_prompt=True, tokenize=False) for sentence in sentences
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)

    # Generate
    output_sequence = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=64,
    )

    # Decode and print the model's responses
    for i in range(batch_size):
        generated_texts = tokenizer.decode(
            output_sequence[i], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print("\033[32m" + f"batch {i} : " + "\033[0m\n" + generated_texts)


if __name__ == "__main__":
    fire.Fire(main)
