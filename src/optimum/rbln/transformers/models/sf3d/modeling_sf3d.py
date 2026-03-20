# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import rebel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import trimesh
from einops import rearrange, repeat
from skimage.measure import marching_cubes

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from .configuration_sf3d import RBLNSF3DForImageTo3DConfig


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

# Accepted inputs for preprocess / generate_scene_codes / image_to_3d (paths, arrays, PIL).
SF3DImageInput = Union[str, Path, np.ndarray, torch.Tensor, PIL.Image.Image]


class _SF3DImageTokenizerWrapper(nn.Module):
    """Wraps DINOv2SingleImageTokenizer with ada-norm modulations for NPU."""

    def __init__(self, image_tokenizer: nn.Module):
        super().__init__()
        self.model = image_tokenizer.model
        self.register_buffer("image_mean", image_tokenizer.image_mean.reshape(1, 3, 1, 1))
        self.register_buffer("image_std", image_tokenizer.image_std.reshape(1, 3, 1, 1))

    def forward(self, pixel_values: torch.Tensor, modulation_cond: torch.Tensor) -> torch.Tensor:
        x = (pixel_values - self.image_mean) / self.image_std
        out = self.model(x, modulation_cond=modulation_cond)
        return out.last_hidden_state.permute(0, 2, 1)


class _SF3DBackboneWrapper(nn.Module):
    """Wraps TwoStreamInterleaveTransformer for NPU."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.backbone(hidden_states, encoder_hidden_states)


class _SF3DPostProcessorWrapper(nn.Module):
    """Wraps PixelShuffleUpsampleNetwork for NPU."""

    def __init__(self, post_processor: nn.Module):
        super().__init__()
        self.post_processor = post_processor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post_processor(x)


class _SF3DTriplaneDecoderWrapper(nn.Module):
    """Combined triplane feature query + decoder MLP heads for NPU.

    Performs grid_sample-based triplane lookup and forwards through
    all MaterialMLP heads in a single compiled graph.
    """

    def __init__(self, decoder_module: nn.ModuleDict, head_order: list, radius: float):
        super().__init__()
        self.radius = radius
        self.head_order = head_order
        for name in head_order:
            self.add_module(f"head_{name}", decoder_module[name])

    def forward(self, positions: torch.Tensor, triplanes: torch.Tensor) -> torch.Tensor:
        pos_norm = positions / self.radius
        features_list = []
        axes_pairs = [[0, 1], [0, 2], [1, 2]]
        for i in range(3):
            plane = triplanes[i : i + 1]
            coords = pos_norm[:, axes_pairs[i]].unsqueeze(0).unsqueeze(0)
            sampled = F.grid_sample(plane, coords, align_corners=True, mode="bilinear")
            features_list.append(sampled.squeeze(2).squeeze(0))
        feats = torch.cat(features_list, dim=0).T

        outputs = []
        for name in self.head_order:
            outputs.append(getattr(self, f"head_{name}")(feats))
        return torch.cat(outputs, dim=-1)


def _make_glb_double_sided(glb_bytes: bytes) -> bytes:
    """Patch a GLB binary to set doubleSided=True on all materials.

    If no materials exist, adds one so WebGL viewers render both face sides.
    """
    magic, version, length = struct.unpack("<III", glb_bytes[:12])
    json_chunk_len = struct.unpack("<I", glb_bytes[12:16])[0]
    json_chunk_type = struct.unpack("<I", glb_bytes[16:20])[0]
    gltf = json.loads(glb_bytes[20 : 20 + json_chunk_len])
    rest = glb_bytes[20 + json_chunk_len :]

    if "materials" not in gltf or len(gltf["materials"]) == 0:
        gltf["materials"] = [
            {
                "pbrMetallicRoughness": {
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                },
                "doubleSided": True,
            }
        ]
        for m in gltf.get("meshes", []):
            for p in m.get("primitives", []):
                if "material" not in p:
                    p["material"] = 0
    else:
        for mat in gltf["materials"]:
            mat["doubleSided"] = True

    new_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(new_json) % 4:
        new_json += b" "

    new_length = 12 + 8 + len(new_json) + len(rest)
    header = struct.pack("<III", magic, version, new_length)
    json_header = struct.pack("<II", len(new_json), json_chunk_type)
    return header + json_header + new_json + rest


def _smooth_vertex_colors(
    colors: np.ndarray,
    faces: np.ndarray,
    n_verts: int,
    iterations: int = 3,
    lam: float = 0.3,
) -> np.ndarray:
    """Laplacian smoothing of per-vertex colors (numpy only, no scipy)."""
    edges_i = np.concatenate([faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 1], faces[:, 2], faces[:, 2]])
    edges_j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 2], faces[:, 0], faces[:, 1]])

    result = colors.astype(np.float32).copy()
    for _ in range(iterations):
        nb_sum = np.zeros_like(result)
        nb_cnt = np.zeros((n_verts, 1), dtype=np.float32)
        np.add.at(nb_sum, edges_i, result[edges_j])
        np.add.at(nb_cnt, edges_i, 1.0)
        nb_cnt[nb_cnt == 0] = 1.0
        result = (1.0 - lam) * result + lam * (nb_sum / nb_cnt)
    return np.clip(result, 0.0, 1.0)


_TRIPLANE_CPU_CHUNK_SIZE = 131072


def _default_cond_c2w(distance: float) -> torch.Tensor:
    return torch.tensor(
        [[0, 0, 1, distance], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )


def _create_intrinsic_from_fov_deg(fov_deg: float, cond_height: int, cond_width: int):
    focal_length = 0.5 * cond_height / np.tan(0.5 * np.deg2rad(fov_deg))
    intrinsic = np.identity(3, dtype=np.float32)
    intrinsic[0, 0] = focal_length
    intrinsic[1, 1] = focal_length
    intrinsic[0, 2] = cond_width / 2.0
    intrinsic[1, 2] = cond_height / 2.0
    intrinsic = torch.from_numpy(intrinsic)
    intrinsic_normed = intrinsic.clone()
    intrinsic_normed[..., 0, 2] /= cond_width
    intrinsic_normed[..., 1, 2] /= cond_height
    intrinsic_normed[..., 0, 0] /= cond_width
    intrinsic_normed[..., 1, 1] /= cond_height
    return intrinsic, intrinsic_normed


def _query_triplane_cpu(positions, triplanes, radius=1.0):
    """CPU triplane query using grid_sample.

    Args:
        positions: [B, N, 3] or [N, 3] query positions in world space
        triplanes: [B, 3, C, H, W] or [3, C, H, W] triplane features
        radius: world-space radius for normalization
    Returns:
        [B, N, 3*C] or [N, 3*C] sampled features
    """
    squeeze_batch = positions.dim() == 2
    if squeeze_batch:
        positions = positions.unsqueeze(0)
        triplanes = triplanes.unsqueeze(0)

    pos_norm = positions / radius
    features_list = []
    axes_pairs = [[0, 1], [0, 2], [1, 2]]
    for plane_idx in range(3):
        plane = triplanes[:, plane_idx]
        coords = pos_norm[..., axes_pairs[plane_idx]]
        grid = coords.unsqueeze(1)
        sampled = F.grid_sample(plane, grid, align_corners=True, mode="bilinear")
        features_list.append(sampled.squeeze(2))
    result = torch.cat(features_list, dim=1).permute(0, 2, 1)

    if squeeze_batch:
        result = result.squeeze(0)
    return result


def preprocess_sf3d_image(
    image: SF3DImageInput,
    size: int = 512,
    background_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    foreground_ratio: float = 0.85,
) -> torch.Tensor:
    """Preprocess image for SF3D: foreground crop (if RGBA), resize, mask blending.

    Caller is responsible for background removal (e.g. rembg) before passing.

    Args:
        image: PIL Image, numpy array, tensor, or file path
        size: target image size (default 512)
        background_color: RGB background fill (default gray 0.5)
        foreground_ratio: foreground-to-frame ratio (default 0.85)
    Returns:
        [1, 1, 3, H, W] tensor ready for SF3D
    """
    if isinstance(image, (str, Path)):
        image = PIL.Image.open(image)

    if isinstance(image, (np.ndarray, torch.Tensor)):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        if image.ndim == 3 and image.shape[0] in (3, 4):
            image = image.transpose(1, 2, 0)
        image = PIL.Image.fromarray(image)

    if image.mode == "RGBA":
        mask_np = np.array(image)[:, :, -1]
        nz_rows = np.flatnonzero(mask_np.any(axis=1))
        nz_cols = np.flatnonzero(mask_np.any(axis=0))
        if len(nz_rows) > 0 and len(nz_cols) > 0:
            x1, x2 = nz_cols[0], nz_cols[-1]
            y1, y2 = nz_rows[0], nz_rows[-1]
            h, w = y2 - y1, x2 - x1
            yc, xc = (y1 + y2) / 2, (x1 + x2) / 2
            scale = max(h, w) / foreground_ratio

            image = tvF.crop(
                image,
                top=int(yc - scale / 2),
                left=int(xc - scale / 2),
                height=int(scale),
                width=int(scale),
            )

    image = image.resize((size, size), PIL.Image.LANCZOS)
    img_np = np.array(image).astype(np.float32) / 255.0

    if img_np.shape[-1] == 4:
        mask = img_np[:, :, 3:4]
        rgb = img_np[:, :, :3]
        bg = np.array(background_color, dtype=np.float32).reshape(1, 1, 3)
        rgb = rgb * mask + bg * (1.0 - mask)
    else:
        rgb = img_np[:, :, :3]

    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1)
    return tensor.unsqueeze(0).unsqueeze(1)


class RBLNSF3DForImageTo3D(RBLNModel):
    """
    RBLN-optimized SF3D (Stable Fast 3D) model for single-image 3D reconstruction.

    It implements the methods to convert a pre-trained SF3D model into a RBLN SF3D model by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    NPU handles the heavy compute (DINOv2 image tokenizer, backbone transformer,
    post-processor) and optionally the triplane query + decoder MLPs. When compiled
    with ``query_chunk_size``, the combined triplane grid_sample + MaterialMLP
    decoder runs as a single NPU graph per chunk. Mesh extraction and texture
    baking remain on CPU due to their dynamic nature.

    **Configuration:**
    This model uses [`RBLNSF3DForImageTo3DConfig`] for configuration. When calling methods like `from_pretrained`,
    the `rbln_config` parameter should be an instance of [`RBLNSF3DForImageTo3DConfig`] or a dictionary
    conforming to its structure.

    Example::

        # Compile
        model = RBLNSF3DForImageTo3D.from_pretrained(
            "stabilityai/stable-fast-3d", export=True,
        )
        model.save_pretrained("SF3D")

        # Inference
        model = RBLNSF3DForImageTo3D.from_pretrained("SF3D")
        mesh = model.image_to_3d("chair.png")
        mesh.export("chair.glb")
    """

    _IMAGE_TOKENIZER_NAME = "image_tokenizer"
    _BACKBONE_NAME = "backbone"
    _POST_PROCESSOR_NAME = "post_processor"
    _TRIPLANE_DECODER_NAME = "triplane_decoder"

    @classmethod
    def get_pytorch_model(
        cls,
        model_id: str,
        use_auth_token: Optional[str] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        rbln_config: Optional[RBLNSF3DForImageTo3DConfig] = None,
        **kwargs,
    ) -> "PreTrainedModel":
        from transformers import PretrainedConfig

        from .sf3d_architecture import SF3D

        model = SF3D.from_pretrained(model_id, config_name="config.yaml", weight_name="model.safetensors")
        model.eval()
        model.config = PretrainedConfig(model_type="sf3d")
        model.dtype = torch.float32
        return model

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNSF3DForImageTo3DConfig) -> nn.Module:
        return _SF3DImageTokenizerWrapper(model.image_tokenizer)

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Any] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNSF3DForImageTo3DConfig] = None,
    ) -> RBLNSF3DForImageTo3DConfig:
        if rbln_config is None:
            rbln_config = RBLNSF3DForImageTo3DConfig()
        if model is None:
            raise ValueError("`model` is required for _update_rbln_config.")

        B = rbln_config.batch_size
        img_size = rbln_config.image_size
        Nv = 1

        dinov2_cfg = model.image_tokenizer.model.config
        patch_size = dinov2_cfg.patch_size
        hidden_size = dinov2_cfg.hidden_size
        n_patches_per_side = img_size // patch_size
        n_img_tokens = n_patches_per_side * n_patches_per_side + 1

        plane_size = model.tokenizer.cfg["plane_size"]
        num_channels = model.tokenizer.cfg["num_channels"]
        n_triplane_tokens = 3 * plane_size * plane_size

        cam_embed_dim = model.camera_embedder.linear.out_features

        compile_cfgs = [
            RBLNCompileConfig(
                compiled_model_name=cls._IMAGE_TOKENIZER_NAME,
                input_info=[
                    ("pixel_values", [B * Nv, 3, img_size, img_size], "float32"),
                    ("modulation_cond", [B * Nv, cam_embed_dim], "float32"),
                ],
            ),
            RBLNCompileConfig(
                compiled_model_name=cls._BACKBONE_NAME,
                input_info=[
                    ("hidden_states", [B, num_channels, n_triplane_tokens], "float32"),
                    ("encoder_hidden_states", [B, Nv * n_img_tokens, hidden_size], "float32"),
                ],
            ),
            RBLNCompileConfig(
                compiled_model_name=cls._POST_PROCESSOR_NAME,
                input_info=[
                    ("x", [B, 3, num_channels, plane_size, plane_size], "float32"),
                ],
            ),
        ]

        if rbln_config.query_chunk_size is not None:
            chunk = rbln_config.query_chunk_size
            with torch.no_grad():
                dummy_tri = torch.zeros(1, 3, num_channels, plane_size, plane_size)
                pp_out = model.post_processor(dummy_tri)
                post_proc_out_channels = pp_out.shape[2]

            compile_cfgs.append(
                RBLNCompileConfig(
                    compiled_model_name=cls._TRIPLANE_DECODER_NAME,
                    input_info=[
                        ("positions", [chunk, 3], "float32"),
                        ("triplanes", [3, post_proc_out_channels, plane_size * 4, plane_size * 4], "float32"),
                    ],
                ),
            )

        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNSF3DForImageTo3DConfig):
        compiled = {}
        cfg_map = {c.compiled_model_name: c for c in rbln_config.compile_cfgs}

        compiled[cls._IMAGE_TOKENIZER_NAME] = cls.compile(
            _SF3DImageTokenizerWrapper(model.image_tokenizer),
            rbln_compile_config=cfg_map[cls._IMAGE_TOKENIZER_NAME],
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device,
        )
        compiled[cls._BACKBONE_NAME] = cls.compile(
            _SF3DBackboneWrapper(model.backbone),
            rbln_compile_config=cfg_map[cls._BACKBONE_NAME],
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device,
        )
        compiled[cls._POST_PROCESSOR_NAME] = cls.compile(
            _SF3DPostProcessorWrapper(model.post_processor),
            rbln_compile_config=cfg_map[cls._POST_PROCESSOR_NAME],
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device,
        )

        if cls._TRIPLANE_DECODER_NAME in cfg_map:
            head_order = [h.name for h in model.decoder.cfg.heads]
            radius = model.cfg.radius
            decoder_module = nn.ModuleDict({name: model.decoder.heads[name] for name in head_order})
            wrapper = _SF3DTriplaneDecoderWrapper(decoder_module, head_order, radius)
            wrapper.eval()
            compiled[cls._TRIPLANE_DECODER_NAME] = cls.compile(
                wrapper,
                rbln_compile_config=cfg_map[cls._TRIPLANE_DECODER_NAME],
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device,
            )

        return compiled

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNSF3DForImageTo3DConfig,
    ) -> None:
        plane_size = model.tokenizer.cfg["plane_size"]
        num_channels = model.tokenizer.cfg["num_channels"]

        with torch.no_grad():
            dummy_tri = torch.zeros(1, 3, num_channels, plane_size, plane_size)
            pp_out = model.post_processor(dummy_tri)
            post_proc_out_channels = pp_out.shape[2]

        decoder_head_specs = []
        for h in model.decoder.cfg.heads:
            decoder_head_specs.append(
                {
                    "name": h.name,
                    "out_channels": h.out_channels,
                    "n_hidden_layers": h.n_hidden_layers,
                    "output_activation": h.output_activation,
                    "out_bias": h.out_bias,
                }
            )

        iso_helper = model.isosurface_helper
        iso_data = {
            "grid_vertices": iso_helper.grid_vertices.cpu(),
            "indices": iso_helper.indices.cpu(),
            "center_indices": iso_helper.center_indices.cpu(),
            "boundary_indices": iso_helper.boundary_indices.cpu(),
            "resolution": iso_helper.resolution,
            "points_range": iso_helper.points_range,
        }

        distance = model.cfg.default_distance
        fovy = model.cfg.default_fovy_deg
        img_size = model.cfg.cond_image_size
        c2w = _default_cond_c2w(distance)
        _, intrinsic_normed = _create_intrinsic_from_fov_deg(fovy, img_size, img_size)
        cond = torch.cat([c2w.reshape(1, -1), intrinsic_normed.reshape(1, -1)], dim=-1)
        with torch.no_grad():
            precomputed_camera_embedding = model.camera_embedder.linear(cond).cpu()

        save_dict = {
            "tokenizer_embeddings": model.tokenizer.embeddings.data.cpu(),
            "tokenizer_config": {
                "plane_size": plane_size,
                "num_channels": num_channels,
            },
            "decoder_config": {
                "in_features": 3 * post_proc_out_channels,
                "n_neurons": model.decoder.cfg.n_neurons,
                "activation": model.decoder.cfg.activation,
                "head_specs": decoder_head_specs,
            },
            "decoder_state_dict": {k: v.cpu() for k, v in model.decoder.state_dict().items()},
            "model_config": {
                "cond_image_size": model.cfg.cond_image_size,
                "default_distance": model.cfg.default_distance,
                "default_fovy_deg": model.cfg.default_fovy_deg,
                "radius": model.cfg.radius,
                "isosurface_resolution": model.cfg.isosurface_resolution,
                "isosurface_threshold": model.cfg.isosurface_threshold,
                "background_color": list(model.cfg.background_color),
            },
            "precomputed_camera_embedding": precomputed_camera_embedding,
            "isosurface_data": iso_data,
        }
        (save_dir_path / subfolder).mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNSF3DForImageTo3DConfig,
    ) -> List[rebel.Runtime]:
        expected = [c.compiled_model_name for c in rbln_config.compile_cfgs]
        if any(n not in rbln_config.device_map for n in expected):
            cls._raise_missing_compiled_file_error(expected)
        return [
            rebel.Runtime(
                compiled_models[i],
                tensor_type="pt",
                device=rbln_config.device_map[name],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for i, name in enumerate(expected)
        ]

    def __post_init__(self, **kwargs) -> None:
        artifacts = torch.load(
            self.model_save_dir / self.subfolder / "torch_artifacts.pth",
            weights_only=False,
            map_location="cpu",
        )
        self.tokenizer_embeddings = artifacts["tokenizer_embeddings"]
        self.tokenizer_config = artifacts["tokenizer_config"]
        self.model_config = artifacts["model_config"]
        self.decoder_config = artifacts["decoder_config"]
        self.isosurface_data = artifacts.get("isosurface_data", None)
        self._camera_embedding = artifacts.get("precomputed_camera_embedding", None)

        self.image_tokenizer_runtime = None
        self.backbone_runtime = None
        self.post_processor_runtime = None
        self.triplane_decoder_runtime = None

        for i, cfg in enumerate(self.rbln_config.compile_cfgs):
            name = cfg.compiled_model_name
            if name == self._IMAGE_TOKENIZER_NAME:
                self.image_tokenizer_runtime = self.model[i]
            elif name == self._BACKBONE_NAME:
                self.backbone_runtime = self.model[i]
            elif name == self._POST_PROCESSOR_NAME:
                self.post_processor_runtime = self.model[i]
            elif name == self._TRIPLANE_DECODER_NAME:
                self.triplane_decoder_runtime = self.model[i]

        self.cpu_decoder = self._load_cpu_decoder(artifacts)
        self.decoder_head_names = [h["name"] for h in self.cpu_decoder["head_specs"]]

        self._npu_head_order = self.decoder_head_names
        self._npu_head_out_channels = [h["out_channels"] for h in self.cpu_decoder["head_specs"]]
        self._npu_query_chunk_size = getattr(self.rbln_config, "query_chunk_size", None)

        return super().__post_init__(**kwargs)

    @staticmethod
    def _load_cpu_decoder(artifacts):
        """Load MaterialMLP decoder heads as CPU nn.Modules."""
        dcfg = artifacts["decoder_config"]
        state_dict = artifacts.get("decoder_state_dict")
        if state_dict is None:
            raise RuntimeError(
                "decoder_state_dict not found in torch_artifacts. Please recompile the model with the latest version."
            )

        head_specs = dcfg["head_specs"]
        in_features = dcfg["in_features"]
        n_neurons = dcfg["n_neurons"]
        activation = dcfg.get("activation", "silu")

        heads = {}
        for hs in head_specs:
            layers = []
            for i in range(hs["n_hidden_layers"]):
                layers.append(nn.Linear(in_features if i == 0 else n_neurons, n_neurons))
                layers.append(nn.SiLU() if activation == "silu" else nn.ReLU())
            layers.append(nn.Linear(n_neurons, hs["out_channels"]))
            heads[hs["name"]] = nn.Sequential(*layers)

        decoder_module = nn.ModuleDict(heads)

        decoder_module.load_state_dict(
            {k.replace("heads.", "", 1): v for k, v in state_dict.items() if k.startswith("heads.")},
            strict=False,
        )
        decoder_module.eval()

        return {"module": decoder_module, "head_specs": head_specs}

    def _get_camera_embedding(self) -> torch.Tensor:
        """Return pre-computed camera embedding (computed at compile time)."""
        if self._camera_embedding is not None:
            return self._camera_embedding
        raise RuntimeError(
            "Pre-computed camera embedding not found in torch_artifacts. "
            "Please recompile the model with the latest version."
        )

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run the full SF3D encoding pipeline to produce triplane scene codes.

        Args:
            pixel_values: [B, Nv, 3, H, W] input images (Nv=1 for single view)
        Returns:
            [B, 3, Cout, Hp, Wp] triplane scene codes
        """
        B, Nv = pixel_values.shape[:2]

        cam_embed = self._get_camera_embedding()
        cam_embed = cam_embed.expand(B * Nv, -1).contiguous()

        pv = rearrange(pixel_values, "B N C H W -> (B N) C H W").contiguous()
        image_tokens = self.image_tokenizer_runtime(pv, cam_embed)
        if isinstance(image_tokens, torch.Tensor):
            image_tokens = rearrange(image_tokens, "(B N) Ct Nt -> B (N Nt) Ct", B=B).contiguous()
        else:
            image_tokens = torch.tensor(image_tokens)
            image_tokens = rearrange(image_tokens, "(B N) Ct Nt -> B (N Nt) Ct", B=B).contiguous()

        tp = repeat(self.tokenizer_embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=B)
        triplane_flat = rearrange(tp, "B Np Ct Hp Wp -> B Ct (Np Hp Wp)").contiguous()

        backbone_out = self.backbone_runtime(triplane_flat, image_tokens)
        if not isinstance(backbone_out, torch.Tensor):
            backbone_out = torch.tensor(backbone_out)

        ps = self.tokenizer_config["plane_size"]
        triplanes = rearrange(backbone_out, "B Ct (Np Hp Wp) -> B Np Ct Hp Wp", Np=3, Hp=ps, Wp=ps).contiguous()

        scene_codes = self.post_processor_runtime(triplanes)
        if not isinstance(scene_codes, torch.Tensor):
            scene_codes = torch.tensor(scene_codes)
        return scene_codes.float()

    def generate_scene_codes(self, image: SF3DImageInput) -> torch.Tensor:
        bg_color = tuple(self.model_config.get("background_color", [0.5, 0.5, 0.5]))
        return self.forward(
            preprocess_sf3d_image(
                image,
                self.model_config["cond_image_size"],
                background_color=bg_color,
            )
        )

    @staticmethod
    def _apply_head_activation(x, activation_name, bias):
        x = x + bias
        if activation_name is None:
            return x
        if activation_name == "sigmoid":
            return torch.sigmoid(x)
        if activation_name == "tanh":
            return torch.tanh(x)
        if activation_name == "trunc_exp":
            return torch.exp(x)
        if activation_name == "normalize_channel_last":
            return F.normalize(x, dim=-1, p=2, eps=1e-7)
        return x

    def _decode_triplane_at_positions(self, positions, scene_code, head_names=None):
        """Query triplane features at positions and decode through MLP heads.

        Uses NPU for triplane query + decoder if the triplane_decoder runtime
        is available (compiled with ``query_chunk_size``). Otherwise falls back
        to CPU execution.
        """
        if head_names is None:
            head_names = self.decoder_head_names

        if self.triplane_decoder_runtime is not None:
            return self._decode_triplane_npu(positions, scene_code, head_names)
        return self._decode_triplane_cpu(positions, scene_code, head_names)

    def _decode_triplane_npu(self, positions, scene_code, head_names):
        """NPU path: triplane query + decoder in a single compiled graph."""
        chunk = self._npu_query_chunk_size or _TRIPLANE_CPU_CHUNK_SIZE
        N = positions.shape[0]
        triplanes = scene_code.squeeze(0).contiguous()  # [3, C, H, W]
        head_spec_map = {hs["name"]: hs for hs in self.cpu_decoder["head_specs"]}

        all_outputs = {h: [] for h in head_names}
        head_offsets = {}
        offset = 0
        for name, nch in zip(self._npu_head_order, self._npu_head_out_channels, strict=True):
            head_offsets[name] = (offset, offset + nch)
            offset += nch

        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            chunk_pos = positions[start:end]

            if chunk_pos.shape[0] < chunk:
                pad = torch.zeros(chunk - chunk_pos.shape[0], 3, device=chunk_pos.device)
                chunk_pos_padded = torch.cat([chunk_pos, pad], dim=0)
            else:
                chunk_pos_padded = chunk_pos

            raw_out = self.triplane_decoder_runtime(chunk_pos_padded.contiguous(), triplanes)
            if not isinstance(raw_out, torch.Tensor):
                raw_out = torch.tensor(raw_out)
            raw_out = raw_out[: end - start]

            for hname in head_names:
                s, e = head_offsets[hname]
                raw = raw_out[:, s:e]
                hs = head_spec_map[hname]
                h_out = self._apply_head_activation(raw, hs["output_activation"], hs["out_bias"])
                all_outputs[hname].append(h_out)

        return {h: torch.cat(v, dim=0) for h, v in all_outputs.items()}

    def _decode_triplane_cpu(self, positions, scene_code, head_names):
        """CPU fallback: triplane query (grid_sample) + decoder MLPs on CPU."""
        radius = self.model_config.get("radius", 0.87)
        chunk = _TRIPLANE_CPU_CHUNK_SIZE
        N = positions.shape[0]
        all_outputs = {h: [] for h in head_names}

        triplanes = scene_code.squeeze(0)  # [3, C, H, W]
        decoder_mod = self.cpu_decoder["module"]
        head_spec_map = {hs["name"]: hs for hs in self.cpu_decoder["head_specs"]}

        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            chunk_pos = positions[start:end]

            feat_input = _query_triplane_cpu(chunk_pos, triplanes, radius)
            feat_input = feat_input.contiguous()

            for hname in head_names:
                raw = decoder_mod[hname](feat_input)
                hs = head_spec_map[hname]
                h_out = self._apply_head_activation(raw, hs["output_activation"], hs["out_bias"])
                all_outputs[hname].append(h_out)

        return {h: torch.cat(v, dim=0) for h, v in all_outputs.items()}

    @torch.no_grad()
    def extract_mesh(
        self,
        scene_codes: torch.Tensor,
        threshold: Optional[float] = None,
        bake_resolution: int = 1024,
        remesh: str = "none",
        vertex_count: int = -1,
    ) -> List[trimesh.Trimesh]:
        """Extract mesh from triplane scene codes using marching cubes.

        Args:
            scene_codes: [B, 3, C, H, W] triplane features
            threshold: isosurface threshold (default from model config)
            bake_resolution: unused (kept for API compatibility)
            remesh: "none", "triangle", or "quad"
            vertex_count: target vertex count for remeshing (-1 = auto)
        Returns:
            list of trimesh.Trimesh meshes with vertex colors
        """
        if threshold is None:
            threshold = self.model_config.get("isosurface_threshold", 10.0)

        radius = self.model_config.get("radius", 0.87)
        resolution = 256

        coords = torch.linspace(-radius, radius, resolution)
        gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
        grid_pos = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

        meshes = []
        for b in range(scene_codes.shape[0]):
            sc = scene_codes[b : b + 1]

            decoded = self._decode_triplane_at_positions(
                grid_pos,
                sc,
                head_names=["density"],
            )
            density = decoded["density"].squeeze(-1)
            level = -(density - threshold)
            level_3d = level.view(resolution, resolution, resolution).cpu().numpy()

            try:
                verts_mc, faces_mc, _, _ = marching_cubes(level_3d, level=0.0)
            except (ValueError, RuntimeError):
                meshes.append(trimesh.Trimesh())
                continue

            verts_np = (verts_mc / (resolution - 1.0) * 2.0 - 1.0) * radius
            verts_np = verts_np.astype(np.float32)
            faces_np = faces_mc.astype(np.int64)

            if remesh in ("triangle", "quad"):
                target = vertex_count if vertex_count > 0 else min(verts_np.shape[0], 10000)
                target_faces = min(target * 2, faces_np.shape[0])
                tmp = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
                try:
                    decimated = tmp.simplify_quadric_decimation(face_count=target_faces)
                    verts_np = decimated.vertices.astype(np.float32)
                    faces_np = decimated.faces.astype(np.int64)
                except Exception as e:
                    logger.warning("Decimation failed (%s), keeping original.", e)

            tex_heads = [h for h in self.decoder_head_names if h not in ("density", "vertex_offset")]
            v_pos = torch.from_numpy(verts_np).float()
            decoded_c = self._decode_triplane_at_positions(v_pos, sc, head_names=tex_heads)

            vertex_colors = None
            albedo = decoded_c.get("features")
            if albedo is not None:
                vc = albedo.detach().cpu().numpy().clip(0.0, 1.0)
                vc = _smooth_vertex_colors(vc, faces_np, verts_np.shape[0], iterations=2, lam=0.15)
                vc_u8 = (vc * 255).astype(np.uint8)
                alpha = np.full((vc_u8.shape[0], 1), 255, dtype=np.uint8)
                vertex_colors = np.concatenate([vc_u8, alpha], axis=1)

            tmesh = trimesh.Trimesh(
                vertices=verts_np,
                faces=faces_np,
                vertex_colors=vertex_colors,
            )

            rot_x = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
            rot_y = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
            tmesh.apply_transform(rot_x)
            tmesh.apply_transform(rot_y)
            meshes.append(tmesh)

        return meshes

    def image_to_3d(
        self,
        image: SF3DImageInput,
        threshold: Optional[float] = None,
        bake_resolution: int = 1024,
        remesh: str = "none",
        vertex_count: int = -1,
    ) -> trimesh.Trimesh:
        """Full pipeline: image -> scene codes -> mesh with vertex colors.

        Args:
            image: PIL Image (RGBA preferred), numpy array, tensor, or file path
            threshold: isosurface threshold (default from model config)
            bake_resolution: unused (kept for API compatibility)
            remesh: "none", "triangle", or "quad" (default "none")
            vertex_count: target vertex count for remeshing (-1 = auto)
        Returns:
            trimesh.Trimesh
        """
        scene_codes = self.generate_scene_codes(image)
        return self.extract_mesh(scene_codes, threshold, bake_resolution, remesh, vertex_count)[0]

    @staticmethod
    def export_mesh(mesh, path: str):
        """Export mesh to file. For GLB, sets doubleSided material for WebGL viewers."""
        mesh.export(path)
        if path.endswith(".glb"):
            with open(path, "rb") as f:
                patched = _make_glb_double_sided(f.read())
            with open(path, "wb") as f:
                f.write(patched)
