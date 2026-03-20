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

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import PIL.Image
import rebel
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from einops import rearrange, reduce, repeat
from skimage.measure import marching_cubes

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from .configuration_triposr import RBLNTripoSRForImageTo3DConfig


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel


class _ImageTokenizerWrapper(nn.Module):
    """Wraps the DINO ViT backbone: normalize + ViT forward -> patch features."""

    def __init__(self, vit_model: nn.Module, image_mean: torch.Tensor, image_std: torch.Tensor):
        super().__init__()
        self.vit_model = vit_model
        self.register_buffer("image_mean", image_mean.reshape(1, 3, 1, 1))
        self.register_buffer("image_std", image_std.reshape(1, 3, 1, 1))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = (pixel_values - self.image_mean) / self.image_std
        out = self.vit_model(x, interpolate_pos_encoding=True)
        return out.last_hidden_state.permute(0, 2, 1)


class _BackboneWrapper(nn.Module):
    """Wraps Transformer1D backbone for cross-attention compilation."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.backbone(hidden_states, encoder_hidden_states=encoder_hidden_states)


class _PostProcessorWrapper(nn.Module):
    """Wraps TriplaneUpsampleNetwork (ConvTranspose2d) for compilation."""

    def __init__(self, upsample: nn.Module):
        super().__init__()
        self.upsample = upsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class _TriplaneQueryWrapper(nn.Module):
    """Wraps triplane query (grid_sample + decoder MLP + activations) for NPU.

    Takes a pre-built sampling grid and triplane, runs grid_sample + NeRF MLP
    decoder + activation functions. Grid construction is done on CPU before
    calling the NPU runtime to avoid dynamic indexing in the compiled graph.

    Input:  grid      [3, chunk_size, 1, 2]  (sampling coordinates in [-1, 1])
            triplane  [3, C, H, W]
    Output: [chunk_size, 4]  (density_act[1], color_rgb[3])
    """

    def __init__(self, decoder_module: nn.Module, renderer_cfg: dict, chunk_size: int):
        super().__init__()
        self.density_bias = renderer_cfg["density_bias"]
        self.use_concat = renderer_cfg["feature_reduction"] == "concat"
        self._density_act_name = renderer_cfg["density_activation"]
        self._color_act_name = renderer_cfg["color_activation"]
        self.chunk_size = chunk_size

        new_layers = []
        for layer in decoder_module.layers:
            if isinstance(layer, (nn.ReLU, nn.SiLU)):
                new_layers.append(type(layer)(inplace=False))
            else:
                new_layers.append(layer)
        self.decoder = nn.Sequential(*new_layers)

    def forward(self, grid: torch.Tensor, triplane: torch.Tensor) -> torch.Tensor:
        out = F.grid_sample(
            triplane,
            grid,
            align_corners=False,
            mode="bilinear",
        )

        out = out.squeeze(3)
        if self.use_concat:
            out = out.permute(2, 0, 1).reshape(self.chunk_size, -1)
        else:
            out = out.mean(dim=0).t()

        raw = self.decoder(out)
        density = raw[:, 0:1] + self.density_bias
        features = raw[:, 1:4]

        if self._density_act_name == "exp":
            density_act = torch.exp(density)
        elif self._density_act_name == "sigmoid":
            density_act = torch.sigmoid(density)
        elif self._density_act_name == "softplus":
            density_act = F.softplus(density)
        else:
            density_act = density

        if self._color_act_name == "sigmoid":
            color = torch.sigmoid(features)
        elif self._color_act_name == "exp":
            color = torch.exp(features)
        else:
            color = features

        return torch.cat([density_act, color], dim=-1)


def _scale_tensor(dat: torch.Tensor, inp_scale: Tuple[float, float], tgt_scale: Tuple[float, float]) -> torch.Tensor:
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def _get_activation(name: str):
    name = name.lower()
    activations = {
        "exp": lambda x: torch.exp(x),
        "sigmoid": lambda x: torch.sigmoid(x),
        "softplus": lambda x: F.softplus(x),
        "none": lambda x: x,
        "linear": lambda x: x,
    }
    if name in activations:
        return activations[name]
    return getattr(F, name)


def _query_triplane(
    decoder: nn.Module,
    positions: torch.Tensor,
    triplane: torch.Tensor,
    radius: float,
    feature_reduction: str,
    density_activation: str,
    density_bias: float,
    color_activation: str,
    chunk_size: int = 65536,
) -> Dict[str, torch.Tensor]:
    input_shape = positions.shape[:-1]
    positions_flat = positions.view(-1, 3)
    positions_norm = _scale_tensor(positions_flat, (-radius, radius), (-1, 1))

    all_density, all_features = [], []

    for i in range(0, max(1, positions_norm.shape[0]), chunk_size):
        chunk = positions_norm[i : i + chunk_size]
        indices2D = torch.stack(
            (chunk[..., [0, 1]], chunk[..., [0, 2]], chunk[..., [1, 2]]),
            dim=-3,
        )
        out = F.grid_sample(
            rearrange(triplane, "Np Cp Hp Wp -> Np Cp Hp Wp", Np=3),
            rearrange(indices2D, "Np N Nd -> Np () N Nd", Np=3),
            align_corners=False,
            mode="bilinear",
        )
        if feature_reduction == "concat":
            out = rearrange(out, "Np Cp () N -> N (Np Cp)", Np=3)
        else:
            out = reduce(out, "Np Cp () N -> N Cp", Np=3, reduction="mean")

        net_out = decoder(out)
        all_density.append(net_out["density"])
        all_features.append(net_out["features"])

    density = torch.cat(all_density, dim=0)
    features = torch.cat(all_features, dim=0)
    density_act = _get_activation(density_activation)(density + density_bias)
    color = _get_activation(color_activation)(features)

    return {
        "density_act": density_act.view(*input_shape, -1),
        "color": color.view(*input_shape, -1),
    }


def _extract_mesh(
    decoder: nn.Module,
    scene_code: torch.Tensor,
    renderer_cfg: dict,
    resolution: int = 256,
    threshold: float = 25.0,
    has_vertex_color: bool = True,
):
    """CPU fallback: extract mesh using CPU decoder."""
    _TRIPLANE_KEYS = {"radius", "feature_reduction", "density_activation", "density_bias", "color_activation", "chunk_size"}
    triplane_kwargs = {k: v for k, v in renderer_cfg.items() if k in _TRIPLANE_KEYS}

    radius = renderer_cfg["radius"]

    x, y, z = (torch.linspace(0, 1, resolution),) * 3
    gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
    grid_vertices = torch.cat([gx.reshape(-1, 1), gy.reshape(-1, 1), gz.reshape(-1, 1)], dim=-1)
    scaled_verts = _scale_tensor(grid_vertices.to(scene_code.device), (0, 1), (-radius, radius))

    with torch.no_grad():
        qr = _query_triplane(decoder, scaled_verts, scene_code, **triplane_kwargs)

    level = -(qr["density_act"].squeeze(-1) - threshold)
    level_3d = level.view(resolution, resolution, resolution).detach().cpu().numpy()

    verts_np, faces_np, _, _ = marching_cubes(level_3d, level=0.0)

    v_pos = torch.from_numpy(verts_np.copy()).float()
    t_pos_idx = torch.from_numpy(faces_np.copy()).long()

    v_pos = v_pos / (resolution - 1.0)
    v_pos = _scale_tensor(v_pos.to(scene_code.device), (0, 1), (-radius, radius))

    color = None
    if has_vertex_color:
        with torch.no_grad():
            qr_c = _query_triplane(decoder, v_pos, scene_code, **triplane_kwargs)
        color = (qr_c["color"].cpu().numpy() * 255).astype(np.uint8)

    return trimesh.Trimesh(
        vertices=v_pos.cpu().numpy(),
        faces=t_pos_idx.cpu().numpy(),
        vertex_colors=color,
    )


def _build_triplane_grid(positions: torch.Tensor, radius: float) -> torch.Tensor:
    """Build grid_sample grid from 3D positions: [N, 3] -> [3, N, 1, 2]."""
    pos_norm = positions / radius
    x, y, z = pos_norm[:, 0:1], pos_norm[:, 1:2], pos_norm[:, 2:3]
    xy = torch.cat([x, y], dim=-1)
    xz = torch.cat([x, z], dim=-1)
    yz = torch.cat([y, z], dim=-1)
    return torch.stack([xy, xz, yz], dim=0).unsqueeze(2)


def _query_npu_chunked(
    runtime,
    positions: torch.Tensor,
    triplane: torch.Tensor,
    chunk_size: int,
    radius: float,
) -> torch.Tensor:
    """Run triplane query on NPU in fixed-size chunks, padding the last chunk."""
    n = positions.shape[0]
    results = []
    for i in range(0, max(1, n), chunk_size):
        chunk_pos = positions[i : i + chunk_size]
        actual_n = chunk_pos.shape[0]
        if actual_n < chunk_size:
            chunk_pos = F.pad(chunk_pos, (0, 0, 0, chunk_size - actual_n))
        grid = _build_triplane_grid(chunk_pos, radius)
        out = runtime(grid, triplane)
        if actual_n < chunk_size:
            out = out[:actual_n]
        results.append(out)
    return torch.cat(results, dim=0)


def _extract_mesh_npu(
    query_runtime,
    scene_code: torch.Tensor,
    renderer_cfg: dict,
    chunk_size: int,
    resolution: int = 256,
    threshold: float = 25.0,
    has_vertex_color: bool = True,
):
    """Extract mesh using NPU triplane_query runtime."""
    radius = renderer_cfg["radius"]

    x, y, z = (torch.linspace(0, 1, resolution),) * 3
    gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
    grid_vertices = torch.cat([gx.reshape(-1, 1), gy.reshape(-1, 1), gz.reshape(-1, 1)], dim=-1)
    scaled_verts = _scale_tensor(grid_vertices, (0, 1), (-radius, radius))

    out = _query_npu_chunked(query_runtime, scaled_verts, scene_code, chunk_size, radius)
    density_act = out[:, 0:1]

    level = -(density_act.squeeze(-1) - threshold)
    level_3d = level.view(resolution, resolution, resolution).detach().cpu().numpy()

    verts_np, faces_np, _, _ = marching_cubes(level_3d, level=0.0)

    v_pos = torch.from_numpy(verts_np.copy()).float()
    t_pos_idx = torch.from_numpy(faces_np.copy()).long()

    v_pos = v_pos / (resolution - 1.0)
    v_pos = _scale_tensor(v_pos, (0, 1), (-radius, radius))

    color = None
    if has_vertex_color:
        color_out = _query_npu_chunked(query_runtime, v_pos, scene_code, chunk_size, radius)
        color = (color_out[:, 1:4].cpu().numpy() * 255).astype(np.uint8)

    return trimesh.Trimesh(
        vertices=v_pos.cpu().numpy(),
        faces=t_pos_idx.cpu().numpy(),
        vertex_colors=color,
    )


def preprocess_image(image, size: int = 512) -> torch.Tensor:
    """Convert PIL/numpy/tensor image to [1, 3, size, size] float32 in [0, 1]."""
    if isinstance(image, PIL.Image.Image):
        image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    elif isinstance(image, np.ndarray):
        image = torch.from_numpy(image.astype(np.float32) if image.dtype == np.uint8 else image.copy())
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.shape[-1] in (3, 4):
        image = image[..., :3].permute(0, 3, 1, 2)
    return F.interpolate(image, (size, size), mode="bilinear", align_corners=False, antialias=True)


class _NeRFMLPDecoder(nn.Module):
    """Lightweight NeRF MLP that outputs density(1) + features(3) = 4 channels."""

    def __init__(self, in_channels: int, n_neurons: int, n_hidden_layers: int, activation: str = "silu"):
        super().__init__()
        act = {"relu": nn.ReLU(inplace=True), "silu": nn.SiLU(inplace=True)}[activation]
        layers = [nn.Linear(in_channels, n_neurons), act]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), act]
        layers.append(nn.Linear(n_neurons, 4))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        inp_shape = x.shape[:-1]
        features = self.layers(x.reshape(-1, x.shape[-1])).reshape(*inp_shape, -1)
        return {"density": features[..., 0:1], "features": features[..., 1:4]}


class RBLNTripoSRForImageTo3D(RBLNModel):
    """
    RBLN-optimized TripoSR model for single-image 3D reconstruction.

    It implements the methods to convert a pre-trained TripoSR model into a RBLN TripoSR model by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    **Configuration:**
    This model uses [`RBLNTripoSRForImageTo3DConfig`] for configuration. When calling methods like `from_pretrained`,
    the `rbln_config` parameter should be an instance of [`RBLNTripoSRForImageTo3DConfig`] or a dictionary
    conforming to its structure.

    Example::

        # Compile
        model = RBLNTripoSRForImageTo3D.from_pretrained(
            "stabilityai/TripoSR", export=True,
        )
        model.save_pretrained("TripoSR")

        # Inference
        model = RBLNTripoSRForImageTo3D.from_pretrained("TripoSR")
        mesh = model.image_to_3d("chair.png")
        mesh.export("chair.obj")
    """

    _IMAGE_TOKENIZER_NAME = "image_tokenizer"
    _BACKBONE_NAME = "backbone"
    _POST_PROCESSOR_NAME = "post_processor"
    _TRIPLANE_QUERY_NAME = "triplane_query"

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
        rbln_config: Optional[RBLNTripoSRForImageTo3DConfig] = None,
        **kwargs,
    ) -> "PreTrainedModel":
        from transformers import PretrainedConfig

        from .triposr_architecture import TSR

        model = TSR.from_pretrained(model_id, config_name="config.yaml", weight_name="model.ckpt")
        model.eval()
        model.config = PretrainedConfig(model_type="triposr")
        model.dtype = torch.float32
        return model

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNTripoSRForImageTo3DConfig) -> nn.Module:
        return _ImageTokenizerWrapper(
            vit_model=model.image_tokenizer.model,
            image_mean=torch.tensor([0.485, 0.456, 0.406]),
            image_std=torch.tensor([0.229, 0.224, 0.225]),
        )

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Any] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNTripoSRForImageTo3DConfig] = None,
    ) -> RBLNTripoSRForImageTo3DConfig:
        if rbln_config is None:
            rbln_config = RBLNTripoSRForImageTo3DConfig()
        if model is None:
            raise ValueError("`model` is required for _update_rbln_config.")

        B = rbln_config.batch_size
        img_size = rbln_config.image_size
        chunk_size = rbln_config.query_chunk_size
        patch_size = 16
        n_tokens = (img_size // patch_size) ** 2 + 1

        plane_size = model.tokenizer.cfg.plane_size
        num_channels = model.tokenizer.cfg.num_channels
        triplane_seq_len = 3 * plane_size * plane_size
        vit_embed_dim = model.image_tokenizer.model.config.hidden_size
        triplane_out_ch = model.post_processor.upsample.out_channels
        triplane_hw = plane_size * 2

        compile_cfgs = [
            RBLNCompileConfig(
                compiled_model_name=cls._IMAGE_TOKENIZER_NAME,
                input_info=[("pixel_values", [B, 3, img_size, img_size], "float32")],
            ),
            RBLNCompileConfig(
                compiled_model_name=cls._BACKBONE_NAME,
                input_info=[
                    ("hidden_states", [B, num_channels, triplane_seq_len], "float32"),
                    ("encoder_hidden_states", [B, n_tokens, vit_embed_dim], "float32"),
                ],
            ),
            RBLNCompileConfig(
                compiled_model_name=cls._POST_PROCESSOR_NAME,
                input_info=[("x", [B * 3, num_channels, plane_size, plane_size], "float32")],
            ),
            RBLNCompileConfig(
                compiled_model_name=cls._TRIPLANE_QUERY_NAME,
                input_info=[
                    ("grid", [3, chunk_size, 1, 2], "float32"),
                    ("triplane", [3, triplane_out_ch, triplane_hw, triplane_hw], "float32"),
                ],
            ),
        ]
        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def get_compiled_model(cls, model: "PreTrainedModel", rbln_config: RBLNTripoSRForImageTo3DConfig):
        compiled: Dict[str, rebel.RBLNCompiledModel] = {}
        cfg_map = {c.compiled_model_name: c for c in rbln_config.compile_cfgs}

        compiled[cls._IMAGE_TOKENIZER_NAME] = cls.compile(
            cls._wrap_model_if_needed(model, rbln_config),
            rbln_compile_config=cfg_map[cls._IMAGE_TOKENIZER_NAME],
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device,
        )
        compiled[cls._BACKBONE_NAME] = cls.compile(
            _BackboneWrapper(model.backbone),
            rbln_compile_config=cfg_map[cls._BACKBONE_NAME],
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device,
        )
        compiled[cls._POST_PROCESSOR_NAME] = cls.compile(
            _PostProcessorWrapper(model.post_processor.upsample),
            rbln_compile_config=cfg_map[cls._POST_PROCESSOR_NAME],
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device,
        )

        renderer_cfg = {
            "radius": model.renderer.cfg.radius,
            "feature_reduction": model.renderer.cfg.feature_reduction,
            "density_activation": model.renderer.cfg.density_activation,
            "density_bias": model.renderer.cfg.density_bias,
            "color_activation": model.renderer.cfg.color_activation,
        }
        query_wrapper = _TriplaneQueryWrapper(
            model.decoder, renderer_cfg, chunk_size=rbln_config.query_chunk_size
        )
        query_wrapper.eval()
        compiled[cls._TRIPLANE_QUERY_NAME] = cls.compile(
            query_wrapper,
            rbln_compile_config=cfg_map[cls._TRIPLANE_QUERY_NAME],
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
        rbln_config: RBLNTripoSRForImageTo3DConfig,
    ) -> None:
        save_dict = {
            "triplane_embeddings": model.tokenizer.embeddings.data.cpu(),
            "triplane_config": {
                "plane_size": model.tokenizer.cfg.plane_size,
                "num_channels": model.tokenizer.cfg.num_channels,
            },
            # Decoder weights kept for backward compat (CPU fallback).
            # When triplane_query NPU runtime is present, these are not loaded.
            "decoder_state_dict": model.decoder.state_dict(),
            "decoder_config": {
                "in_channels": model.decoder.cfg.in_channels,
                "n_neurons": model.decoder.cfg.n_neurons,
                "n_hidden_layers": model.decoder.cfg.n_hidden_layers,
                "activation": model.decoder.cfg.activation,
            },
            "renderer_config": {
                "radius": model.renderer.cfg.radius,
                "feature_reduction": model.renderer.cfg.feature_reduction,
                "density_activation": model.renderer.cfg.density_activation,
                "density_bias": model.renderer.cfg.density_bias,
                "color_activation": model.renderer.cfg.color_activation,
                "num_samples_per_ray": model.renderer.cfg.num_samples_per_ray,
            },
            "cond_image_size": model.cfg.cond_image_size,
            "query_chunk_size": rbln_config.query_chunk_size,
        }
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNTripoSRForImageTo3DConfig,
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

    def _assign_runtimes(self) -> None:
        for i, cfg in enumerate(self.rbln_config.compile_cfgs):
            name = cfg.compiled_model_name
            if name == self._IMAGE_TOKENIZER_NAME:
                self.image_tokenizer_runtime = self.model[i]
            elif name == self._BACKBONE_NAME:
                self.backbone_runtime = self.model[i]
            elif name == self._POST_PROCESSOR_NAME:
                self.post_processor_runtime = self.model[i]
            elif name == self._TRIPLANE_QUERY_NAME:
                self.triplane_query_runtime = self.model[i]

    def _load_decoder(self, artifacts: dict) -> nn.Module:
        cfg = artifacts["decoder_config"]
        decoder = _NeRFMLPDecoder(
            in_channels=cfg["in_channels"],
            n_neurons=cfg["n_neurons"],
            n_hidden_layers=cfg["n_hidden_layers"],
            activation=cfg["activation"],
        )
        original_sd = artifacts["decoder_state_dict"]
        new_sd = {"layers." + k if not k.startswith("layers.") else k: v for k, v in original_sd.items()}
        decoder.load_state_dict(new_sd)
        decoder.eval()
        return decoder

    def __post_init__(self, **kwargs) -> None:
        artifacts = torch.load(
            self.model_save_dir / self.subfolder / "torch_artifacts.pth",
            weights_only=False,
            map_location="cpu",
        )
        self.triplane_embeddings = artifacts["triplane_embeddings"]
        self.triplane_config = artifacts["triplane_config"]
        self.renderer_config = artifacts["renderer_config"]
        self.cond_image_size = artifacts["cond_image_size"]
        self.query_chunk_size = artifacts.get("query_chunk_size", 65536)

        self.image_tokenizer_runtime = None
        self.backbone_runtime = None
        self.post_processor_runtime = None
        self.triplane_query_runtime = None
        self._assign_runtimes()

        if self.triplane_query_runtime is not None:
            self.decoder = None
            logger.info("triplane_query NPU runtime loaded; CPU decoder skipped.")
        else:
            self.decoder = self._load_decoder(artifacts)
            logger.info("triplane_query NPU runtime not found; using CPU decoder fallback.")

        return super().__post_init__(**kwargs)

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate triplane scene codes from input images.

        Args:
            pixel_values: [B, 3, H, W] float32 images in [0, 1].

        Returns:
            scene_codes: [B, 3, C_out, H_out, W_out] triplane features.
        """
        B = pixel_values.shape[0]
        image_tokens = rearrange(self.image_tokenizer_runtime(pixel_values), "B C N -> B N C")

        triplane_tokens = rearrange(
            repeat(self.triplane_embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=B),
            "B Np Ct Hp Wp -> B Ct (Np Hp Wp)",
        )

        tokens = self.backbone_runtime(triplane_tokens, image_tokens)

        ps = self.triplane_config["plane_size"]
        triplanes = rearrange(tokens, "B Ct (Np Hp Wp) -> (B Np) Ct Hp Wp", Np=3, Hp=ps, Wp=ps)
        scene_codes_flat = self.post_processor_runtime(triplanes)
        return rearrange(scene_codes_flat, "(B Np) Co Hp Wp -> B Np Co Hp Wp", B=B, Np=3)

    def generate_scene_codes(self, image) -> torch.Tensor:
        """Preprocess a single image and return scene codes."""
        return self.forward(preprocess_image(image, self.cond_image_size))

    def extract_mesh(
        self,
        scene_codes: torch.Tensor,
        resolution: int = 256,
        threshold: float = 25.0,
        has_vertex_color: bool = True,
    ) -> list:
        """Extract 3D meshes from scene codes.

        Uses NPU triplane_query runtime when available, falls back to CPU decoder.

        Returns:
            List of trimesh.Trimesh objects.
        """
        meshes = []
        for i in range(scene_codes.shape[0]):
            if self.triplane_query_runtime is not None:
                mesh = _extract_mesh_npu(
                    self.triplane_query_runtime,
                    scene_codes[i],
                    self.renderer_config,
                    self.query_chunk_size,
                    resolution,
                    threshold,
                    has_vertex_color,
                )
            else:
                mesh = _extract_mesh(
                    self.decoder,
                    scene_codes[i],
                    self.renderer_config,
                    resolution,
                    threshold,
                    has_vertex_color,
                )
            meshes.append(mesh)
        return meshes

    def image_to_3d(self, image, resolution: int = 256, threshold: float = 25.0):
        """End-to-end: single image -> 3D mesh with vertex colors."""
        return self.extract_mesh(self.generate_scene_codes(image), resolution, threshold)[0]
