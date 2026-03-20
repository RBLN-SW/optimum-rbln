# SF3D (Stable Fast 3D) — RBLN Integration

## Overview

SF3D is a single-image 3D mesh reconstruction model from Stability AI.
Given one image it produces a 3D mesh with vertex colors.

Original repository: [Stability-AI/stable-fast-3d](https://github.com/Stability-AI/stable-fast-3d)

## Architecture

### Pipeline

```
Image ──► DINOv2 (Image Tokenizer) ──► TwoStreamInterleaveTransformer (Backbone)
  │                                            │
  │   camera embedding ────────────────────────┘
  │
  └──► PixelShuffleUpsampleNetwork (Post-Processor) ──► Triplane Scene Codes
                                                              │
                                                              ▼
                                                    Triplane Query + Decoder
                                                     (density + color)
                                                              │
                                                              ▼
                                                       Marching Cubes
                                                              │
                                                              ▼
                                                    Vertex-Colored Mesh (GLB)
```

### NPU vs CPU


| Component                            | Runs on                          | Notes                                                                                                                |
| ------------------------------------ | -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| DINOv2 Image Tokenizer               | **NPU**                          | Includes ada-norm modulation                                                                                         |
| TwoStreamInterleaveTransformer       | **NPU**                          | Backbone transformer                                                                                                 |
| PixelShuffleUpsampleNetwork          | **NPU**                          | Post-processor                                                                                                       |
| Triplane Query + MaterialMLP Decoder | **NPU**                          | Combined grid_sample + MLP; chunked at `query_chunk_size`. Falls back to CPU if compiled without `query_chunk_size`. |
| Marching Cubes                       | CPU                              | `skimage.measure.marching_cubes` — variable output shapes                                                            |
| Vertex Coloring                      | **NPU** (query) + CPU (assembly) | Triplane query at mesh vertices for RGB                                                                              |


#### Why Marching Cubes cannot run on NPU

NPU requires **static computation graphs** with fixed tensor shapes at compile time.
Marching cubes produces a variable number of vertices and faces depending on the isosurface:

- Input: fixed `[resolution³]` density grid
- Output: variable `[V, 3]` vertices and `[F, 3]` faces

This is fundamentally incompatible with static-graph NPU compilation.

### Coloring: Vertex Colors

The triplane decoder has multiple heads. For mesh extraction, only `density` (geometry)
and `features` (albedo RGB) are used:


| Head       | Purpose                                      |
| ---------- | -------------------------------------------- |
| `density`  | SDF for marching cubes isosurface extraction |
| `features` | Albedo / basecolor (RGB, sigmoid activation) |


Vertex colors are obtained by querying the triplane at each mesh vertex position,
followed by lightweight Laplacian smoothing (numpy-only, no external dependencies).

### Key Files


| File                    | Role                                                                              |
| ----------------------- | --------------------------------------------------------------------------------- |
| `sf3d_architecture.py`  | Model definitions — DINOv2 subclasses (modulation), backbone, decoder, isosurface |
| `modeling_sf3d.py`      | NPU compilation wrappers, runtime management, inference pipeline                  |
| `configuration_sf3d.py` | `RBLNSF3DForImageTo3DConfig`                                                      |


### Compilation Wrappers (modeling_sf3d.py)


| Wrapper                       | Compiled Name      | Input → Output                                         |
| ----------------------------- | ------------------ | ------------------------------------------------------ |
| `_SF3DImageTokenizerWrapper`  | `image_tokenizer`  | `[B, 3, H, W]` + `[B, D]` → `[B, C, N]`                |
| `_SF3DBackboneWrapper`        | `backbone`         | triplane tokens + image tokens → updated tokens        |
| `_SF3DPostProcessorWrapper`   | `post_processor`   | `[B, 3, C, H, W]` → `[B, 3, C', H', W']`               |
| `_SF3DTriplaneDecoderWrapper` | `triplane_decoder` | `[N, 3]` + `[3, C, H, W]` → `[N, 10]` (raw MLP output) |


## Usage

```python
from optimum.rbln import RBLNSF3DForImageTo3D

# Compile
model = RBLNSF3DForImageTo3D.from_pretrained(
    "stabilityai/stable-fast-3d", export=True,
)
model.save_pretrained("SF3D")

# Inference
model = RBLNSF3DForImageTo3D.from_pretrained("SF3D")
mesh = model.image_to_3d("chair.png")
model.export_mesh(mesh, "chair.glb")
```

### Configuration

```python
from optimum.rbln import RBLNSF3DForImageTo3DConfig

config = RBLNSF3DForImageTo3DConfig(
    batch_size=1,
    image_size=512,
    query_chunk_size=131072,  # NPU triplane decoder chunk size (None for CPU-only)
)
```

### Dependencies


| Package                                          | With optimum-rbln          | Purpose                                       |
| ------------------------------------------------ | -------------------------- | --------------------------------------------- |
| `torch`, `torchvision`, `transformers`           | Yes                        | Core ML framework                             |
| `einops`, `trimesh`, `scikit-image`, `omegaconf` | Yes                        | SF3D / TripoSR (mesh, marching cubes, config) |
| `safetensors`                                    | Yes (via transformers)     | Weight loading                                |
| `rembg`                                          | No (inference script only) | Background removal preprocessing              |


