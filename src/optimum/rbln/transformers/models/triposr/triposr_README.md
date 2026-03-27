# TripoSR — RBLN Integration

## Overview

TripoSR is a single-image 3D reconstruction model from VAST AI Research and Stability AI.
Given one image it produces a 3D mesh with vertex colors.

Original repository: [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR)

## Architecture

### Pipeline

```
Image ──► ViT (Image Tokenizer) ──► Transformer1D (Backbone)
                                          │
                                          ▼
                                  ConvTranspose2d (Post-Processor)
                                          │
                                          ▼
                                   Triplane Scene Codes
                                          │
                                          ▼
                              Triplane Query + NeRF MLP
                              (grid_sample + decoder)
                                          │
                                          ▼
                                    Marching Cubes
                                          │
                                          ▼
                                   Mesh (OBJ/GLB)
```

### NPU vs CPU

| Component | Runs on | Notes |
|---|---|---|
| ViT Image Tokenizer | **NPU** | `_ImageTokenizerWrapper` |
| Transformer1D Backbone | **NPU** | `_BackboneWrapper` |
| ConvTranspose2d Post-Processor | **NPU** | `_PostProcessorWrapper` |
| Triplane Query + NeRF MLP | **NPU** | `_TriplaneQueryWrapper` (grid_sample + MLP + activations) |
| Marching Cubes | CPU | `skimage.measure.marching_cubes` — variable output shapes |
| Vertex Coloring | **NPU** (query) + CPU (assembly) | Triplane query at mesh vertices for RGB |

#### Why Marching Cubes cannot run on NPU

NPU requires **static computation graphs** with fixed tensor shapes at compile time.
Marching cubes produces a variable number of vertices and faces depending on the isosurface:

- Input: fixed `[resolution³]` density grid
- Output: variable `[V, 3]` vertices and `[F, 3]` faces

This is fundamentally incompatible with static-graph NPU compilation.

### Key Files

| File | Role |
|---|---|
| `triposr_architecture.py` | Model definitions — self-contained port from TripoSR repo |
| `modeling_triposr.py` | NPU compilation wrappers, runtime management, inference pipeline |
| `configuration_triposr.py` | `RBLNTripoSRForImageTo3DConfig` |

### Compilation Wrappers (modeling_triposr.py)

| Wrapper | Compiled Name | Input → Output |
|---|---|---|
| `_ImageTokenizerWrapper` | `image_tokenizer` | `[B, 3, H, W]` → `[B, C, N]` |
| `_BackboneWrapper` | `backbone` | triplane tokens + image tokens → updated tokens |
| `_PostProcessorWrapper` | `post_processor` | `[B×3, C, H, W]` → `[B×3, C', H', W']` |
| `_TriplaneQueryWrapper` | `triplane_query` | `[3, chunk, 1, 2]` + `[3, C, H, W]` → `[chunk, 4]` |

### Inference Modes

**With NPU triplane_query** (default):
- Image → NPU encoding pipeline → scene codes
- Mesh extraction uses NPU triplane query for both geometry and color
- CPU decoder is not loaded

**Without NPU triplane_query** (backward-compatible):
- Image → NPU encoding pipeline → scene codes
- Mesh extraction uses CPU NeRF MLP decoder fallback
- Decoder weights loaded from `torch_artifacts.pth`

## Usage

```python
from optimum.rbln import RBLNTripoSRForImageTo3D

# Compile
model = RBLNTripoSRForImageTo3D.from_pretrained(
    "stabilityai/TripoSR", export=True,
)
model.save_pretrained("TripoSR")

# Inference
model = RBLNTripoSRForImageTo3D.from_pretrained("TripoSR")
mesh = model.image_to_3d("chair.png")
mesh.export("chair.obj")
```

### Configuration

```python
from optimum.rbln import RBLNTripoSRForImageTo3DConfig

config = RBLNTripoSRForImageTo3DConfig(
    batch_size=1,
    image_size=512,
    query_chunk_size=65536,  # positions per NPU triplane query call
)
```

### Dependencies

| Package | With optimum-rbln | Purpose |
|---|---|---|
| `torch`, `torchvision`, `transformers` | Yes | Core ML framework |
| `einops`, `trimesh`, `scikit-image`, `omegaconf` | Yes | SF3D / TripoSR (mesh, marching cubes, config) |
| `rembg` | No (inference script only) | Background removal preprocessing |
