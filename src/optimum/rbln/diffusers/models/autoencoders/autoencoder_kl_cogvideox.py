# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from typing import TYPE_CHECKING

import rebel
import torch
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import (
    AutoencoderKLCogVideoX,
    CogVideoXCausalConv3d,
    CogVideoXDecoder3D,
)
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from rebel.compile_context import CompileContext

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ....utils.logging import get_logger
from ...configurations import RBLNAutoencoderKLCogVideoXConfig
from .vae import RBLNRuntimeCogVideoXVAEDecoder, RBLNRuntimeCogVideoXVAEEncoder


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

logger = get_logger(__name__)


# frame (D) axis of the (n, c, d, h*w) cache, for rbln_cache_update
_CACHE_FRAME_AXIS = 2


def _resnet_cache_convs(prefix: tuple[str, ...], resnets: torch.nn.ModuleList, scale: int) -> list:
    convs = []
    for j, resnet in enumerate(resnets):
        if getattr(resnet, "use_conv_shortcut", False) and isinstance(
            getattr(resnet, "conv_shortcut", None), CogVideoXCausalConv3d
        ):
            raise NotImplementedError("CogVideoX resnets with a causal conv shortcut are not supported.")
        for name in ("conv1", "conv2"):
            convs.append(((*prefix, f"resnet_{j}", name), getattr(resnet, name), scale))
    return convs


def get_cache_convs(decoder: CogVideoXDecoder3D) -> list[tuple[tuple[str, ...], CogVideoXCausalConv3d, int]]:
    """
    List the causal convs of `CogVideoXDecoder3D.forward` that carry a temporal cache, in a fixed order.

    The Wan VAE keeps its caches as a constant table. Here they come from the module tree, so a reduced
    config (fewer blocks or channels) gets the right table too. Each entry is
    `(conv_cache key path, conv module, spatial scale over the latent)`.
    Kernel-1 convs (the spatial-norm `conv_y` / `conv_b`) ignore their cache, so they are left out.
    """
    convs = [(("conv_in",), decoder.conv_in, 1)]
    convs += _resnet_cache_convs(("mid_block",), decoder.mid_block.resnets, 1)
    scale = 1
    for i, up_block in enumerate(decoder.up_blocks):
        convs += _resnet_cache_convs((f"up_block_{i}",), up_block.resnets, scale)
        if up_block.upsamplers is not None:
            scale *= 2
    convs.append((("conv_out",), decoder.conv_out, scale))

    for path, conv, _ in convs:
        if conv.pad_mode == "replicate" or conv.time_kernel_size < 2:
            raise NotImplementedError(f"Unexpected CogVideoX causal conv at {'.'.join(path)}.")
    return convs


def get_cache_specs(decoder: CogVideoXDecoder3D) -> list[tuple[tuple[str, ...], int, int, int]]:
    """`(conv_cache key path, channels, depth, spatial scale over the latent)` for each cached conv."""
    return [
        (path, conv.conv.in_channels, conv.time_kernel_size - 1, scale)
        for path, conv, scale in get_cache_convs(decoder)
    ]


def get_cache_size(
    decoder: CogVideoXDecoder3D, latent_height: int, latent_width: int, batch_size: int = 1
) -> list[list[int]]:
    return [
        [batch_size, channels, depth, latent_height * scale, latent_width * scale]
        for _, channels, depth, scale in get_cache_specs(decoder)
    ]


# The last axis of a static cache must be a multiple of 64 (DRAM rows are 128B aligned; with h*w = 5400
# a (1, 512, 2, 5400) cache update fails the partition check with "wrong shape for custom op").
# 480x720 has 60x90 and 120x180 caches, so each cache gets one of these layouts:
#   "flat":    (n, c, d, h*w)  when h*w is a multiple of 64 (the 240x360 and 480x720 caches)
#   "chlast":  (n, h*w, d, c)  otherwise, with c a multiple of 64: written through a real permute
#                              (n, c, d, h, w) -> (n, h, w, d, c), like the Qwen3.5 conv state (B, K-1, C)
# A plain reshape to (n, h*w, d, c) without the permute does not compile: it re-cuts the 128B-padded
# w rows into c-wide rows, and the full decoder then aborts in FoldMemoryTransfer.
_CACHE_ALIGN = 64


def _cache_layout(c: int, h: int, w: int) -> str:
    if (h * w) % _CACHE_ALIGN == 0:
        return "flat"
    if c % _CACHE_ALIGN == 0:
        return "chlast"
    raise NotImplementedError(
        f"A {c}-channel conv cache at {h}x{w} cannot be stored on the device: either the channel count or "
        f"h*w (the latent resolution times the block's upscale) must be a multiple of {_CACHE_ALIGN}."
    )


def _cache_input_shape(shape) -> list:
    # per-slot (n,c,d,h,w) -> input_info shape of the static cache.
    n, c, d, h, w = shape
    if _cache_layout(c, h, w) == "chlast":
        return [n, h * w, d, c]
    return [n, c, d, h * w]


def _to_cache(x: torch.Tensor) -> torch.Tensor:
    n, c, d, h, w = x.shape
    if _cache_layout(c, h, w) == "chlast":
        return x.permute(0, 3, 4, 2, 1).reshape(n, h * w, d, c).contiguous()
    return x.reshape(n, c, d, h * w).contiguous()


def _from_cache(cache: torch.Tensor, c: int, h: int, w: int) -> torch.Tensor:
    n, _, d, _ = cache.shape
    if _cache_layout(c, h, w) == "chlast":
        return cache.reshape(n, h, w, d, c).permute(0, 4, 3, 1, 2)
    return cache.reshape(n, c, d, h, w)


def _get_path(tree: dict, path: Sequence[str]) -> torch.Tensor:
    for key in path:
        tree = tree[key]
    return tree


def _set_path(tree: dict, path: Sequence[str], value: torch.Tensor) -> None:
    for key in path[:-1]:
        tree = tree.setdefault(key, {})
    tree[path[-1]] = value


class _VAECogVideoXDecoderBase(torch.nn.Module):
    def __init__(self, vae: AutoencoderKLCogVideoX, latent_height: int, latent_width: int):
        super().__init__()
        self.decoder = vae.decoder
        convs = get_cache_convs(self.decoder)
        self.cache_paths = [path for path, _, _ in convs]
        self.cache_hw = [(latent_height * scale, latent_width * scale) for _, _, scale in convs]
        self.cache_channels = [conv.conv.in_channels for _, conv, _ in convs]

    def _decode(self, z: torch.Tensor, conv_cache: dict | None, caches: Sequence[torch.Tensor]):
        """
        Run the HF decoder, then write each new conv cache (idx 1..) to its static buffer.

        All writes stay at the end of the graph, as in the Wan VAE. Writing each cache right after its
        conv (to free the slices early) corrupts the decoder_n output on the device: the new cache does
        not depend on the old one that the same conv read, so the write can run before that read.
        """
        out, new_cache = self.decoder(z, conv_cache=conv_cache)
        position = torch.tensor(0, dtype=torch.int16)
        axis = torch.tensor(_CACHE_FRAME_AXIS, dtype=torch.int16)
        dummy_outs = []
        for path, cache in zip(self.cache_paths[1:], caches, strict=True):
            item = _to_cache(_get_path(new_cache, path)).to(cache.dtype)
            dummy_outs.append(torch.ops.rbln_custom_ops.rbln_cache_update(cache, item, position, axis))
        # idx 0 (conv_in cache, a slice of z): runtime output
        return out, _get_path(new_cache, self.cache_paths[0]).contiguous(), dummy_outs


class _VAECogVideoXDecoder0(_VAECogVideoXDecoderBase):
    """Wrapper module for the first CogVideoX VAE decoder chunk (no incoming conv cache)."""

    def forward(self, z, *caches) -> torch.Tensor:
        return self._decode(z, None, caches)


class _VAECogVideoXDecoderN(_VAECogVideoXDecoderBase):
    """Wrapper module for the later CogVideoX VAE decoder chunks (reads the previous conv cache)."""

    def forward(self, z, cache_0, *caches) -> torch.Tensor:
        conv_cache = {}
        _set_path(conv_cache, self.cache_paths[0], cache_0)
        for path, c, (h, w), cache in zip(
            self.cache_paths[1:], self.cache_channels[1:], self.cache_hw[1:], caches, strict=True
        ):
            _set_path(conv_cache, path, _from_cache(cache, c, h, w).to(z.dtype))
        return self._decode(z, conv_cache, caches)


class _VAECogVideoXEncoder(torch.nn.Module):
    """Wrapper module for the CogVideoX VAE encoder on a 1-frame image (image-to-video).

    diffusers encodes fewer than `num_sample_frames_batch_size` (8) frames in one call with no conv cache,
    so the image is one graph without static caches.
    """

    def __init__(self, vae: AutoencoderKLCogVideoX):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(x, conv_cache=None)
        if self.quant_conv is not None:
            h = self.quant_conv(h)
        return h


class RBLNAutoencoderKLCogVideoX(RBLNModel):
    """
    RBLN implementation of AutoencoderKLCogVideoX for diffusion models.

    This model is used to accelerate AutoencoderKLCogVideoX models from diffusers library on RBLN NPUs.
    Like the Wan VAE, the decoder's temporal conv caches stay on the device as static buffers and the frame
    loop runs on the host: `decoder_0` decodes the first latent chunk without a cache and `decoder_n` decodes
    each later chunk with the previous cache. With `uses_encoder` (image-to-video), `encoder` encodes a
    1-frame image.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    auto_model_class = AutoencoderKLCogVideoX
    hf_library_name = "diffusers"
    _rbln_config_class = RBLNAutoencoderKLCogVideoXConfig

    @classmethod
    def _reconstruct_model_if_needed(cls, model: "PreTrainedModel"):
        if model.dtype != torch.float32:
            logger.warning(
                f"RBLNAutoencoderKLCogVideoX was handed a {model.dtype} module; casting it to float32 for compilation."
            )
            model = model.to(torch.float32)
        return super()._reconstruct_model_if_needed(model)

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.image_size = self.rbln_config.image_size
        self.use_slicing = self.rbln_config.use_slicing
        self.use_tiling = False
        self.num_latent_frames_batch_size = self.rbln_config.num_latent_frames_batch_size

        # By index: without runtimes, self.model is an UnavailableRuntime (len 0, any index returns itself).
        runtimes = {cfg.compiled_model_name: self.model[i] for i, cfg in enumerate(self.rbln_config.compile_cfgs)}
        self.decoder = RBLNRuntimeCogVideoXVAEDecoder(
            runtime=runtimes["decoder_0"],
            decoder_n=runtimes.get("decoder_n"),
            main_input_name="z",
            first_chunk_size=self.rbln_config.first_chunk_size,
            frame_batch_size=self.rbln_config.num_latent_frames_batch_size,
            dtype=self.rbln_config.dtype,
        )
        if "encoder" in runtimes:
            self.encoder = RBLNRuntimeCogVideoXVAEEncoder(
                runtime=runtimes["encoder"], main_input_name="x", dtype=self.rbln_config.dtype
            )
        else:
            self.encoder = None

    @classmethod
    def _needs_decoder_n(cls, rbln_config: RBLNAutoencoderKLCogVideoXConfig) -> bool:
        return rbln_config.num_latent_frames > rbln_config.first_chunk_size

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNAutoencoderKLCogVideoXConfig
    ) -> torch.nn.Module:
        latent_h = rbln_config.height // rbln_config.vae_scale_factor_spatial
        latent_w = rbln_config.width // rbln_config.vae_scale_factor_spatial
        models = [_VAECogVideoXDecoder0(model, latent_h, latent_w).eval()]
        if cls._needs_decoder_n(rbln_config):
            models.append(_VAECogVideoXDecoderN(model, latent_h, latent_w).eval())
        if rbln_config.uses_encoder:
            models.append(_VAECogVideoXEncoder(model).eval())
        return tuple(models)

    @classmethod
    def get_compiled_model(
        cls, model, rbln_config: RBLNAutoencoderKLCogVideoXConfig
    ) -> dict[str, rebel.RBLNCompiledModel]:
        models = dict(
            zip(
                [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs],
                cls._wrap_model_if_needed(model, rbln_config),
                strict=True,
            )
        )

        # D0 and DN wrap the SAME vae.decoder -> share its weights on device (use_weight_sharing).
        context = CompileContext(use_weight_sharing=True)
        context, example_inputs = cls.get_dec_compile_cfg(context, rbln_config)

        # Graphs of one CompileSession reuse an earlier graph's intermediate buffer when it is large
        # enough, and only then. decoder_n needs the largest one (7.25 GiB vs 4.40 GiB at 480x720,
        # 49 frames), so compile it first and let decoder_0 (and the encoder) reuse it; the other order
        # allocates several buffers and does not fit on one device.
        order = [name for name in ("decoder_n", "decoder_0", "encoder") if name in models]
        cfgs = {cfg.compiled_model_name: cfg for cfg in rbln_config.compile_cfgs}
        compiled = {}
        for name in order:
            compiled[name] = cls.compile(
                models[name],
                rbln_compile_config=cfgs[name],
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device_map[name],
                example_inputs=example_inputs[name],
                compile_context=context,
            )
        return {cfg.compiled_model_name: compiled[cfg.compiled_model_name] for cfg in rbln_config.compile_cfgs}

    @classmethod
    def get_dec_compile_cfg(cls, context, rbln_config):
        cfgs = {cfg.compiled_model_name: cfg for cfg in rbln_config.compile_cfgs}
        example_inputs = {}
        dec0_example_inputs = cfgs["decoder_0"].get_dummy_inputs(fill=0)
        static_tensors = {}
        for (name, _, _), tensor in zip(cfgs["decoder_0"].input_info, dec0_example_inputs, strict=True):
            if "feat_cache" in name:
                static_tensors[name] = tensor
                context.mark_static_address(tensor)
        example_inputs["decoder_0"] = dec0_example_inputs

        if "decoder_n" in cfgs:
            decn_example_inputs = cfgs["decoder_n"].get_dummy_inputs(fill=0, static_tensors=static_tensors)
            for (name, _, _), tensor in zip(cfgs["decoder_n"].input_info, decn_example_inputs, strict=True):
                if ("feat_cache" in name) and ("feat_cache_0" not in name):
                    context.mark_static_address(tensor)
            example_inputs["decoder_n"] = decn_example_inputs
        if "encoder" in cfgs:
            example_inputs["encoder"] = cfgs["encoder"].get_dummy_inputs(fill=0)
        return context, example_inputs

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        vae_scale_factor_spatial = pipe.vae_scale_factor_spatial
        if rbln_config.vae.height is None:
            rbln_config.vae.height = pipe.transformer.config.sample_height * vae_scale_factor_spatial
        if rbln_config.vae.width is None:
            rbln_config.vae.width = pipe.transformer.config.sample_width * vae_scale_factor_spatial
        if rbln_config.vae.num_frames is None:
            rbln_config.vae.num_frames = pipe.transformer.config.sample_frames

        rbln_config.vae.num_channels_latents = pipe.transformer.config.out_channels
        rbln_config.vae.vae_scale_factor_temporal = pipe.vae_scale_factor_temporal
        rbln_config.vae.vae_scale_factor_spatial = vae_scale_factor_spatial
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor] | None,
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNAutoencoderKLCogVideoXConfig,
    ) -> RBLNAutoencoderKLCogVideoXConfig:
        missing = [key for key in ("height", "width", "num_frames") if getattr(rbln_config, key) is None]
        if missing:
            raise ValueError(
                f"{', '.join(f'`{key}`' for key in missing)} must be specified to compile the CogVideoX VAE. "
                "Compiling through a pipeline fills them with that pipeline's defaults; a standalone "
                "compile must pass them in rbln_config."
            )

        if rbln_config.num_channels_latents is None:
            rbln_config.num_channels_latents = model_config.latent_channels
        if rbln_config.vae_scale_factor_temporal is None:
            rbln_config.vae_scale_factor_temporal = model_config.temporal_compression_ratio
        if rbln_config.vae_scale_factor_spatial is None:
            rbln_config.vae_scale_factor_spatial = 2 ** (len(model_config.block_out_channels) - 1)

        latent_height = rbln_config.height // rbln_config.vae_scale_factor_spatial
        latent_width = rbln_config.width // rbln_config.vae_scale_factor_spatial
        batch_size = rbln_config.batch_size
        z_channels = model_config.latent_channels
        cache_shapes = get_cache_size(model.decoder, latent_height, latent_width, batch_size)

        def z_info(num_latent_frames):
            return ("z", [batch_size, z_channels, num_latent_frames, latent_height, latent_width], rbln_config.dtype)

        dec_0_input_info = [z_info(rbln_config.first_chunk_size)]
        dec_n_input_info = [
            z_info(rbln_config.num_latent_frames_batch_size),
            ("feat_cache_0", list(cache_shapes[0]), rbln_config.dtype),
        ]
        # The static caches stay on the device, so their dtype is free: bf16 halves ~5.3 GiB of fp32
        # buffers at 480x720, which decoder_0 + decoder_n need to fit on one device.
        for i, shape in enumerate(cache_shapes[1:], start=1):
            cache_shape = _cache_input_shape(shape)
            dec_0_input_info.append((f"feat_cache_{i}", cache_shape, rbln_config.cache_dtype))
            dec_n_input_info.append((f"feat_cache_{i}", cache_shape, rbln_config.cache_dtype))

        compile_cfgs = [RBLNCompileConfig(compiled_model_name="decoder_0", input_info=dec_0_input_info)]
        if cls._needs_decoder_n(rbln_config):
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name="decoder_n", input_info=dec_n_input_info))
        if rbln_config.uses_encoder:
            # Image-to-video encodes one image as a 1-frame video.
            enc_input_info = [
                (
                    "x",
                    [batch_size, model_config.in_channels, 1, rbln_config.height, rbln_config.width],
                    rbln_config.dtype,
                )
            ]
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name="encoder", input_info=enc_input_info))

        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: list[rebel.RBLNCompiledModel],
        rbln_config: RBLNAutoencoderKLCogVideoXConfig,
    ) -> list[rebel.Runtime]:
        names = [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs]
        return [
            rebel.Runtime(
                compiled_model,
                tensor_type="pt",
                device=rbln_config.device_map[name],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for compiled_model, name in zip(compiled_models, names, strict=True)
        ]

    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        """
        Encode a 1-frame image (image-to-video) into a latent distribution.

        Args:
            x: The image to encode, shaped (batch, channels, 1, height, width).
            return_dict:
                Whether to return output as a dictionary. Defaults to True.

        Returns:
            The latent distribution or AutoencoderKLOutput if return_dict=True
        """
        if self.encoder is None:
            raise ValueError("The VAE was compiled without the encoder; compile it with `uses_encoder=True`.")
        if self.use_slicing and x.shape[0] > 1:
            h = torch.cat([self.encoder.encode(x_slice) for x_slice in x.split(1)])
        else:
            h = self.encoder.encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> torch.Tensor | DecoderOutput:
        """
        Decode a latent representation into a video.

        Args:
            z: The latent representation to decode, shaped (batch, channels, frames, height, width).
            return_dict:
                Whether to return output as a dictionary. Defaults to True.

        Returns:
            The decoded video or DecoderOutput if return_dict=True
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded = torch.cat([self.decoder.decode(z_slice) for z_slice in z.split(1)])
        else:
            decoded = self.decoder.decode(z)

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)


__all__ = [
    "RBLNAutoencoderKLCogVideoX",
]
