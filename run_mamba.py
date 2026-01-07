import os
import typing

import fire
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from optimum.rbln import RBLNMamba2ForCausalLM
from optimum.rbln.transformers.models.mamba2 import RBLNMamba2ForCausalLMConfig


def _pearsonr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = (torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)).clamp_min(1e-12)
    return float((x @ y / denom).item())


def main(
    model_id: str = "AntonV/mamba2-130m-hf",
    batch_size: int = 1,
    from_transformers: bool = False,
    max_seq_len: typing.Optional[int] = None,
    diff: bool = False,
    n_layers: typing.Optional[int] = None,
    compiled_model_path: typing.Optional[str] = None,
    max_new_tokens: int = 10,
    padding_side: str = "right",
    rbln_create_runtimes: bool = False,
):
    target_config = AutoConfig.from_pretrained(model_id)
    if n_layers is not None:
        target_config.num_hidden_layers = int(n_layers)

    if compiled_model_path is None:
        compiled_model_path = os.path.basename(model_id)

    # Tokenize early so we can infer max_seq_len if needed.
    prompts = [
        "Hey how are you doing?",
        "Name the largest country in the world.",
    ]
    prompts = [prompts[i % len(prompts)] for i in range(batch_size)]

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side=padding_side)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    if max_seq_len is None:
        max_seq_len = int(inputs["input_ids"].shape[1])

    if from_transformers:
        rbln_config = RBLNMamba2ForCausalLMConfig(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            create_runtimes=rbln_create_runtimes,
        )
        model = RBLNMamba2ForCausalLM.from_pretrained(
            model_id,
            export=True,
            rbln_config=rbln_config,
            config=target_config,
        )
        model.save_pretrained(compiled_model_path)
    else:
        model = RBLNMamba2ForCausalLM.from_pretrained(compiled_model_path, export=False)

    rbln_outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_logits=True,
    )
    rbln_sequences = rbln_outputs.sequences
    rbln_logits = rbln_outputs.logits

    if diff:
        golden_model = AutoModelForCausalLM.from_pretrained(model_id, config=target_config)
        golden_outputs = golden_model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_logits=True,
        )
        golden_logits = golden_outputs.logits

        for step, (r, g) in enumerate(zip(rbln_logits, golden_logits)):
            print(f"step {step}: pearson={_pearsonr_torch(r.detach(), g.detach()):.6f}")
            for b in range(r.shape[0]):
                print(
                    f"  step {step}, batch {b}: pearson={_pearsonr_torch(r[b].detach(), g[b].detach()):.6f}"
                )

    print("RBLN Sequence")
    for i in range(batch_size):
        generated_text = tokenizer.decode(rbln_sequences[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(f"batch {i}:\n{generated_text}")


if __name__ == "__main__":
    fire.Fire(main)