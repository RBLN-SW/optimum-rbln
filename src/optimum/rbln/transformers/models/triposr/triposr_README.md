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
| Marching Cubes | CPU | (see below) |
| Vertex Coloring | **NPU** (query) + CPU (assembly) | (see below) |

#### Why Marching Cubes cannot run on NPU

NPU requires **static computation graphs** with fixed input/output tensor shapes at compile time.
`skimage.measure.marching_cubes` produces a variable number of vertices and faces
depending on the isosurface geometry:

- Input: fixed `[resolution³]` density grid
- Output: variable `[V, 3]` vertices and `[F, 3]` faces, where V and F depend on the surface

This is fundamentally incompatible with static-graph NPU compilation.
Additionally, marching cubes is implemented in C/Cython (`skimage`), not as a PyTorch graph.

#### Vertex Coloring — already uses NPU

Vertex coloring queries the triplane at mesh vertex positions to obtain RGB values.
In the NPU path (`_extract_mesh_npu`), this is done via `_query_npu_chunked`:

1. Vertex positions are split into **fixed-size chunks** (padded if needed)
2. Each chunk is sent to the **NPU `triplane_query` runtime** (grid_sample + MLP + sigmoid)
3. The output `[chunk_size, 4]` contains `[density, R, G, B]`; color channels `[:, 1:4]` are extracted
4. Results are concatenated and trimmed to actual vertex count on CPU

The heavy computation (triplane sampling + NeRF MLP) runs on NPU.
Only the final assembly (trimming, uint8 conversion, mesh construction) is on CPU.

### Key Files

| File | Role |
|---|---|
| `triposr_architecture.py` | Model definitions — self-contained port from TripoSR repo |
| `modeling_triposr.py` | NPU compilation wrappers, runtime management, inference pipeline |
| `configuration_triposr.py` | `RBLNTripoSRForImageTo3DConfig` |

### triposr_architecture.py Convention

All model components are self-contained (no external `TripoSR` dependency at runtime).
The file includes ViT tokenizer, Transformer1D backbone, NeRF MLP decoder, renderer,
and Marching Cubes helper — ported from the original TripoSR repository.

### Compilation Wrappers (modeling_triposr.py)

Each NPU-compiled subgraph has a thin `nn.Module` wrapper:

| Wrapper | Compiled Name | Input → Output |
|---|---|---|
| `_ImageTokenizerWrapper` | `image_tokenizer` | `[B, 3, H, W]` → `[B, C, N]` |
| `_BackboneWrapper` | `backbone` | triplane tokens + image tokens → updated tokens |
| `_PostProcessorWrapper` | `post_processor` | `[B×3, C, H, W]` → `[B×3, C', H', W']` |
| `_TriplaneQueryWrapper` | `triplane_query` | `[3, 1, chunk, 2]` + `[3, C, H, W]` → `[chunk, 4]` |

The `_TriplaneQueryWrapper` combines `grid_sample` + NeRF MLP decoder + activation
functions into a single NPU runtime. This differs from SF3D where the triplane query
and decoder heads are separate runtimes.

### Inference Modes

**With NPU triplane_query** (default for newly compiled models):
- Image → NPU encoding pipeline → scene codes
- Mesh extraction: `_extract_mesh_npu()` uses NPU triplane query for both geometry and color
- CPU NeRF MLP decoder is not loaded

**Without NPU triplane_query** (backward-compatible):
- Image → NPU encoding pipeline → scene codes
- Mesh extraction: `_extract_mesh()` uses CPU NeRF MLP decoder
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
    batch_size=1,            # batch size
    image_size=512,          # input image resolution
    query_chunk_size=65536,  # positions per NPU triplane query call
)
```
