# Compile: 명시적 config만 사용 (preprocessor 불필요)
# - rbln_image_size: (height, width), default (1008, 1008)
# - rbln_max_text_len: 77 (CLIP)
# - rbln_example_inputs 생략 시 내부에서 dummy 생성
from transformers import Sam3Processor
from optimum.rbln import RBLNSam3Model
import torch
from PIL import Image
import requests

# 1. Compile (preprocessor 없이, image_size만 명시)
model = RBLNSam3Model.from_pretrained(
    "facebook/sam3",
    export=True,
    rbln_create_runtimes=False,
    rbln_image_size=(1008, 1008),
    rbln_max_text_len=77,
)
model.save_pretrained("sam3")

# 2. Load
model = RBLNSam3Model.from_pretrained("sam3", export=False)

# 3. Inference (preprocessor 사용)
image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
processor = Sam3Processor.from_pretrained("facebook/sam3")
inputs = processor(images=image, text="ear", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]


n = len(results["masks"])
print(f"Found {n} object(s)\n")
for i in range(n):
    box = results["boxes"][i].tolist()
    score = results["scores"][i].item()
    mask_shape = tuple(results["masks"][i].shape)
    print(f"  [{i+1}] score: {score:.3f}  box(xyxy): {box}  mask: {mask_shape}")
# Results contain:
# - masks: Binary masks resized to original image size
# - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
# - scores: Confidence scores