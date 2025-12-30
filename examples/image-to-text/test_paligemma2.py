import requests
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from optimum.rbln import RBLNPaliGemmaForConditionalGeneration


model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-mix-224")
processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")

prompt = "Where is the cat standing?"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text=prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(
    **inputs,
)

rbln_model = RBLNPaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma2-3b-mix-224", export=True, rbln_tensor_parallel_size=4
)
rbln_generate_ids = rbln_model.generate(**inputs)
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
print(processor.batch_decode(rbln_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
