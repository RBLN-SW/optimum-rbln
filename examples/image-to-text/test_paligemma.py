
from optimum.rbln import RBLNPaliGemmaForConditionalGeneration


# model_id = "google/paligemma-3b-mix-224"
model_id = "google/paligemma2-3b-mix-224"

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
# image = Image.open(requests.get(url, stream=True).raw)

# model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
# processor = AutoProcessor.from_pretrained(model_id)

# # Instruct the model to create a caption in Spanish
# prompt = "caption es"
# model_inputs = processor(text=prompt, images=image, return_tensors="pt")
# input_len = model_inputs["input_ids"].shape[-1]

# with torch.inference_mode():
#     generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
#     generation = generation[0][input_len:]
#     decoded = processor.decode(generation, skip_special_tokens=True)

rbln_model = RBLNPaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    export=True,
    rbln_config={
        "batch_size": 2,
        "language_model": {
            "prefill_chunk_size": 8192,
        },
    },
    rbln_tensor_parallel_size=4,
)
# rbln_generation = rbln_model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
# rbln_generation = rbln_generation[0][input_len:]
# rbln_decoded = processor.decode(rbln_generation, skip_special_tokens=True)
# print(decoded)
# print(rbln_decoded)
