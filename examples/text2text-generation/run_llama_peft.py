import os
import typing

import fire
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizerFast

from optimum.rbln import RBLNLlamaForCausalLM


def main(
    model_id: str = "meta-llama/Meta-Llama-3-8B",
    batch_size: int = 1,
    from_transformers: bool = False,
    max_seq_len: typing.Optional[int] = None,
    tensor_parallel_size: typing.Optional[int] = 4,
    use_inputs_embeds: bool = None,
):
    if from_transformers:
        # Get pretrained hf model
        model = LlamaForCausalLM.from_pretrained(model_id)

        # Merge lora weights
        model = PeftModel.from_pretrained(model, "FinGPT/fingpt-mt_llama3-8b_lora")
        model = model.merge_and_unload()
        model = model.eval()

        # Compile from lora meged model
        model = RBLNLlamaForCausalLM.from_model(
            model,
            rbln_batch_size=batch_size,
            rbln_max_seq_len=max_seq_len,
            rbln_tensor_parallel_size=tensor_parallel_size,
            rbln_use_inputs_embeds=use_inputs_embeds,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        # Load compiled model
        model = RBLNLlamaForCausalLM.from_pretrained(model_id=os.path.basename(model_id), export=False)

    # Prepare inputs
    tokenizer = LlamaTokenizerFast.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Make prompts
    prompt = [
        """Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: FINANCING OF ASPOCOMP 'S GROWTH Aspocomp is aggressively pursuing its growth strategy by increasingly focusing on technologically more demanding HDI printed circuit boards PCBs .
Answer: """,
        """Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .
Answer: """,
    ] * ((batch_size + 1) // 2)
    prompt = prompt[:-1] if batch_size % 2 == 1 else prompt
    tokens = tokenizer(prompt, return_tensors="pt", padding=True, max_length=512)

    # Generate
    res = model.generate(**tokens, max_length=512)

    # Decode and print the model's responses
    res_sentences = [tokenizer.decode(i) for i in res]
    out_text = [o.split("Answer: ")[1] for o in res_sentences]
    for sentiment in out_text:
        print(sentiment)


if __name__ == "__main__":
    fire.Fire(main)
