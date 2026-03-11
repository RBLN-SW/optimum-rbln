import os
import typing

import fire
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.rbln import RBLNAutoModelForCausalLM


def generate_output(model, model_id, batch_size):
        # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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
    
    output= model.generate(**tokens, do_sample=False, max_new_tokens=64, return_dict_in_generate=True, output_logits=True)
    
    res = output.sequences
    logits = output.logits

    # Decode and print the model's responses
    res_sentences = [tokenizer.decode(i) for i in res]
    out_text = [o.split("Answer: ")[1] for o in res_sentences]
            
    return out_text, logits

def compare_output(native_out_text, native_logits, rbln_out_text, rbln_logits):
    if native_out_text is not None:
        for i in range(len(native_out_text)):
            print(f"Native out text {i}: {native_out_text[i]}")
    
    if rbln_out_text is not None:
        for i in range(len(rbln_out_text)):
            print(f"Rbln out text {i}: {rbln_out_text[i]}")

    if native_logits is not None and rbln_logits is not None:
        for i in range(len(native_logits)):
            print(f"Native logits {i}: {native_logits[i]}")
            print(f"Rbln logits {i}: {rbln_logits[i]}")
            print("--------------------------------")
            
        for batch_idx, (nl, rl) in enumerate(zip(native_logits.detach().numpy(), rbln_logits.detach().numpy())):
            for token_idx, (n, r) in enumerate(zip(nl, rl)):
                corr = np.corrcoef(n, r)
                print(f"Batch {batch_idx}, Token {token_idx}:")
                print(f"Native logit: {n}")
                print(f"Rbln logit: {r}")
                print(f"Correlation: {corr[0, 1]}")

def main(
    model_id: str = "skt/A.X-K1",
    batch_size: int = 1,
    compile: bool = False,
    layers: typing.Optional[int] = None,
    max_seq_len: typing.Optional[int] = None,
    tensor_parallel_size: typing.Optional[int] = 4,
    use_inputs_embeds: bool = None,
    native_run: bool = False,
    rbln_run: bool = False,
):
    if compile:
        # Get pretrained hf model
        rbln_model = RBLNAutoModelForCausalLM.from_pretrained(model_id, rbln_batch_size=batch_size, rbln_max_seq_len=max_seq_len, rbln_tensor_parallel_size=tensor_parallel_size, num_hidden_layers=layers, trust_remote_code=True)
        model.save_pretrained(os.path.basename(model_id))
    elif rbln_run:
        # Load compiled model
        rbln_model = RBLNAutoModelForCausalLM.from_pretrained(model_id=os.path.basename(model_id), export=False)
    
    native_out_text = None
    native_logits = None
    rbln_out_text = None
    rbln_logits = None

    if native_run:
        model = AutoModelForCausalLM.from_pretrained(model_id, num_hidden_layers=layers, trust_remote_code=True)
        native_out_text, native_logits = generate_output(model, model_id, batch_size)
    
    if rbln_run:
        rbln_out_text, rbln_logits = generate_output(rbln_model, model_id, batch_size)        

    compare_output(native_out_text, native_logits, rbln_out_text, rbln_logits)



if __name__ == "__main__":
    fire.Fire(main)
