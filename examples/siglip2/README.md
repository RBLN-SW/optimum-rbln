# SigLIP 2 NaFlex

The native NaFlex image/text model preserves per-image aspect ratios and patch masks. The compiled interface takes the processor's patchified images, spatial shapes and token IDs. Image and text batch sizes are independent. The checkpoint defaults are 256 image patches and 64 text positions.

```python
from PIL import Image
from transformers import AutoProcessor
from optimum.rbln import RBLNSiglip2Model

checkpoint = "google/siglip2-base-patch16-naflex"
revision = "b53b807d3a2d5e2b3911292f2d69e5341cdc064c"
processor = AutoProcessor.from_pretrained(checkpoint, revision=revision)
model = RBLNSiglip2Model.from_pretrained(
    checkpoint,
    revision=revision,
    export=True,
    rbln_batch_size=1,
    rbln_text_batch_size=2,
)
inputs = processor(
    images=Image.open("image.jpg").convert("RGB"),
    text=["a photo of a cat", "a photo of a dog"],
    padding="max_length",
    max_length=64,
    truncation=True,
    return_tensors="pt",
)
outputs = model(**inputs)
print(outputs.logits_per_image.sigmoid())
model.save_pretrained("siglip2-rbln")
```

Token and patch tensor shapes must match the compiled capacities. Change `rbln_max_seq_len`, `rbln_max_patches` or the batch settings before compilation to use different capacities. Shorter text tensors are rejected because padding them after tokenization would change the native last-token pooling semantics. Use the native processor with `padding="max_length"`.

Every positive image grid fitting the patch budget is supported, including different grids within one batch. Exact native frozen position encodings are packed into a lookup table. At the native budget of 256 patches, the table contains 205,109 positions from 1,466 grids, requiring 630,094,848 bytes before compiler packing. Runtime spatial metadata selects the correct positions and repeats the first position for padding. The returned fields include normalized image/text embeddings, calibrated similarity logits and both pooled encoder outputs. Training loss and extra attention/hidden-state diagnostics are not compiled.

This implementation is under numerical validation. The initial CR13 run passes logits and normalized embeddings, but the returned vision hidden state exceeds the predeclared NRMSE threshold. Do not treat compilation or CPU equivalence as completed hardware acceptance. Follow [the tracking issue](https://github.com/rebellions-sw/rebel_compiler/issues/14883) for current compiler and CI results.
