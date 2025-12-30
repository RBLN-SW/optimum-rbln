import torch
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor

from optimum.rbln import RBLNColPaliForRetrieval


model_name = "vidore/colpali-v1.3-hf"

model = ColPaliForRetrieval.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
).eval()

processor = ColPaliProcessor.from_pretrained(model_name, padding_side="left")

# Your inputs
images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (16, 16), color="black"),
]
queries = [
    "What is the organizational structure for our R&D department?",
    "Can you provide a breakdown of last year’s financial performance?",
]

# Process the inputs
batch_images = processor(images=images).to(model.device)
batch_queries = processor(text=queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

import pdb


pdb.set_trace()


# Score the queries against the images
scores = processor.score_retrieval(query_embeddings.embeddings, image_embeddings.embeddings)


rbln_model = RBLNColPaliForRetrieval.from_pretrained(
    # model_name,
    model_id="rbln_colpali",
    # export=True,
    export=False,
    # rbln_tensor_parallel_size=4,
    # rbln_batch_size=2,
    # rbln_config={
    #     "vlm": {
    #         "language_model": {"prefill_chunk_size" : 8192},
    #         "output_hidden_states": True,
    #     }
    # },
)
# rbln_model.save_pretrained("rbln_colpali")

rbln_image_embeddings = rbln_model(**batch_images)
rbln_query_embeddings = rbln_model(**batch_queries)

rbln_scores = processor.score_retrieval(rbln_query_embeddings.embeddings, rbln_image_embeddings.embeddings)

print(scores)
print(rbln_scores)
