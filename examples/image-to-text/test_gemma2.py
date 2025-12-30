from transformers import AutoModelForCausalLM, AutoTokenizer

from optimum.rbln import RBLNGemma2ForCausalLM


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    # torch_dtype=torch.bfloat16,
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_new_tokens=32)

rbln_model = RBLNGemma2ForCausalLM.from_pretrained("google/gemma-2-2b-it", export=True, rbln_tensor_parallel_size=4)
rbln_model.save_pretrained("gemma2-2b-it")
rbln_outputs = rbln_model.generate(**input_ids, max_new_tokens=32)

print(tokenizer.decode(outputs[0]))
print(tokenizer.decode(rbln_outputs[0]))
