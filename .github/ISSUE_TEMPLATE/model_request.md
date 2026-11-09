---
name: 🎯 New Model Support Request
about: Request support for a new model in optimum-rbln
title: '[MODEL] '
labels: 'model-request'
assignees: ''
---

### Model Information
- Model name:
- Model link (HuggingFace/Paper/GitHub):
- Task type (e.g., Text Generation, Image Classification):

### Model Details
- Input modality (Text/Image/Audio/Other):
- Output modality (Text/Image/Audio/Other):
- Model size (parameters):
- Required memory (if known):

### Input/Output Tensor Information
- Can input/output shapes be inferred from model config? Yes/No
- If yes, please explain the relationship:
  ```
  Example:
  - Input shape: (batch_size, sequence_length, hidden_size)
    - sequence_length from config.max_position_embeddings
    - hidden_size from config.hidden_size
  - Output shape: (batch_size, sequence_length, vocab_size)
    - vocab_size from config.vocab_size
  ```
- If no, please specify:
  - Input tensor shapes:
  - Output tensor shapes:
  - Dynamic dimensions (if any):

### Usage Example
Please provide a minimal example of how the model is typically used:
```python
# Example code showing basic model usage
```

### Popularity & Impact
Please help us understand the importance and potential impact of this model. You can include metrics such as:
- Number of downloads/usage statistics (if available)
- Industry adoption examples
- Community interest indicators
- Any other relevant information demonstrating the model's significance
