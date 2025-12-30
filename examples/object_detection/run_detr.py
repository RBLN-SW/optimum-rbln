import requests
import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

from optimum.rbln import RBLNDetrForObjectDetection


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]


rbln_model = RBLNDetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm", export=True, rbln_image_size=(800, 1066))
rbln_model.save_pretrained("detr-resnet-50-optimum")
rbln_outputs = rbln_model(inputs.pixel_values)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")

print("--------------------------------")


rbln_results = processor.post_process_object_detection(rbln_outputs, target_sizes=target_sizes, threshold=0.9)[0]
for score, label, box in zip(rbln_results["scores"], rbln_results["labels"], rbln_results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {rbln_model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )

import numpy as np
from scipy import stats


logits_res = stats.pearsonr(outputs.logits.detach().numpy().reshape(-1), rbln_outputs.logits.reshape(-1)).statistic
print(f"logits_pearson_corr : {logits_res}")
pred_boxes_res = stats.pearsonr(
    outputs.pred_boxes.detach().numpy().reshape(-1), rbln_outputs.pred_boxes.reshape(-1)
).statistic
print(f"pred_boxes_pearson_corr : {pred_boxes_res}")
last_hidden_state_res = stats.pearsonr(
    outputs.last_hidden_state.detach().numpy().reshape(-1), rbln_outputs.last_hidden_state.reshape(-1)
).statistic
print(f"last_hidden_state_pearson_corr : {last_hidden_state_res}")
encoder_last_hidden_state_res = stats.pearsonr(
    outputs.encoder_last_hidden_state.detach().numpy().reshape(-1), rbln_outputs.encoder_last_hidden_state.reshape(-1)
).statistic
print(f"encoder_last_hidden_state_pearson_corr : {encoder_last_hidden_state_res}")

logits_diff = outputs.logits - rbln_outputs.logits
pred_boxes_diff = outputs.pred_boxes - rbln_outputs.pred_boxes
last_hidden_state_diff = outputs.last_hidden_state - rbln_outputs.last_hidden_state
encoder_last_hidden_state_diff = outputs.encoder_last_hidden_state - rbln_outputs.encoder_last_hidden_state
logits_l1_diff = (np.abs(logits_diff.detach().numpy())).max()
pred_boxes_l1_diff = (np.abs(pred_boxes_diff.detach().numpy())).max()
last_hidden_state_l1_diff = (np.abs(last_hidden_state_diff.detach().numpy())).max()
encoder_last_hidden_state_l1_diff = (np.abs(encoder_last_hidden_state_diff.detach().numpy())).max()

print(f"logits_l1_diff : {logits_l1_diff}")
print(f"pred_boxes_l1_diff : {pred_boxes_l1_diff}")
print(f"last_hidden_state_l1_diff : {last_hidden_state_l1_diff}")
print(f"encoder_last_hidden_state_l1_diff : {encoder_last_hidden_state_l1_diff}")
