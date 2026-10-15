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
from dataclasses import dataclass
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
from ...configurations import RBLNAutoencoderKLCogVideoXConfig
from .vae import RBLNRuntimeCogVideoXVAEDecoder, RBLNRuntimeCogVideoXVAEEncoder


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


# Blocks that run at 1x and 2x the latent resolution. rbln_cache_update needs the last axis of a static cache
# to be a multiple of 64 elements (128 bytes), which h*w is not there at 480x720 (5400 and 21600), so their
# caches are stored channels last: (n, h*w, d, c). At 4x and 8x, h*w is 16x and 64x the latent h*w, which is
# aligned for even latent sizes (the transformer's 2x2 patches need them), so those caches stay (n, c, d, h*w).
_CHANNELS_LAST_BLOCKS = ("mid_block", "up_block_0", "up_block_1")


@dataclass(frozen=True)
class _ConvCache:
    """
    The temporal cache of one causal conv in the decoder: the last `depth` input frames of that conv.

    `path` is its key path in the nested `conv_cache` dict of `CogVideoXDecoder3D.forward`, e.g.
    ("up_block_0", "resnet_1", "conv2"). diffusers keeps it as (n, c, d, h, w); on the device it is stored as
    (n, c, d, h*w), or as (n, h*w, d, c) for the `_CHANNELS_LAST_BLOCKS`. The channels-last layout needs a real
    permute: a plain reshape of (n, c, d, h, w) to that shape aborts the compile in FoldMemoryTransfer.
    """

    path: tuple[str, ...]
    channels: int
    depth: int
    height: int
    width: int

    @property
    def channels_last(self) -> bool:
        return self.path[0] in _CHANNELS_LAST_BLOCKS

    def shape(self, batch_size: int) -> list[int]:
        return [batch_size, self.channels, self.depth, self.height, self.width]

    def compile_shape(self, batch_size: int) -> list[int]:
        hw = self.height * self.width
        if self.channels_last:
            return [batch_size, hw, self.depth, self.channels]
        return [batch_size, self.channels, self.depth, hw]

    def to_cache(self, x: torch.Tensor) -> torch.Tensor:
        n, c, d, h, w = x.shape
        if self.channels_last:
            return x.permute(0, 3, 4, 2, 1).reshape(n, h * w, d, c).contiguous()
        return x.reshape(n, c, d, h * w).contiguous()

    def from_cache(self, cache: torch.Tensor) -> torch.Tensor:
        n, c, d, h, w = cache.shape[0], self.channels, self.depth, self.height, self.width
        if self.channels_last:
            return cache.reshape(n, h, w, d, c).permute(0, 4, 3, 1, 2)
        return cache.reshape(n, c, d, h, w)

    def get(self, conv_cache: dict) -> torch.Tensor:
        for key in self.path:
            conv_cache = conv_cache[key]
        return conv_cache

    def put(self, conv_cache: dict, value: torch.Tensor) -> None:
        for key in self.path[:-1]:
            conv_cache = conv_cache.setdefault(key, {})
        conv_cache[self.path[-1]] = value


def _decoder_conv_caches(decoder: CogVideoXDecoder3D, latent_height: int, latent_width: int) -> list[_ConvCache]:
    """
    The convs of `CogVideoXDecoder3D.forward` that carry a temporal cache, in a fixed order.

    The list comes from the module tree, so reduced configs (fewer blocks or channels) work too. Kernel-1 convs
    (the spatial-norm `conv_y` / `conv_b`) ignore their cache and are left out.
    """
    convs = [(("conv_in",), decoder.conv_in, 1)]

    def add_resnets(block_name: str, resnets: torch.nn.ModuleList, scale: int) -> None:
        for j, resnet in enumerate(resnets):
            if getattr(resnet, "use_conv_shortcut", False) and isinstance(
                getattr(resnet, "conv_shortcut", None), CogVideoXCausalConv3d
            ):
                raise NotImplementedError("CogVideoX resnets with a causal conv shortcut are not supported.")
            convs.extend(
                ((block_name, f"resnet_{j}", name), getattr(resnet, name), scale) for name in ("conv1", "conv2")
            )

    add_resnets("mid_block", decoder.mid_block.resnets, 1)
    scale = 1
    for i, up_block in enumerate(decoder.up_blocks):
        add_resnets(f"up_block_{i}", up_block.resnets, scale)
        if up_block.upsamplers is not None:
            scale *= 2
    convs.append((("conv_out",), decoder.conv_out, scale))

    caches = []
    for path, conv, scale in convs:
        if conv.pad_mode == "replicate" or conv.time_kernel_size < 2:
            raise NotImplementedError(f"Unexpected CogVideoX causal conv at {'.'.join(path)}.")
        depth = conv.time_kernel_size - 1
        caches.append(_ConvCache(path, conv.conv.in_channels, depth, latent_height * scale, latent_width * scale))
    return caches


class _VAECogVideoXDecoder(torch.nn.Module):
    """
    One chunk of `AutoencoderKLCogVideoX._decode` as a graph.

    The conv caches carry frames from one chunk to the next. All but the first stay on the device as static
    buffers (graph inputs `conv_cache_1`, ...), updated with `rbln_cache_update`. The first one, of `conv_in`,
    holds latent frames (16 channels at 60x90 for 480x720), which fit neither device layout, so the graph
    returns it and the runtime passes it to the next chunk as `conv_cache_0`.
    """

    def __init__(self, vae: AutoencoderKLCogVideoX, latent_height: int, latent_width: int):
        super().__init__()
        self.decoder = vae.decoder
        caches = _decoder_conv_caches(self.decoder, latent_height, latent_width)
        self.conv_in_cache = caches[0]
        self.static_caches = caches[1:]

    def _decode(self, z: torch.Tensor, conv_cache: dict | None, buffers: Sequence[torch.Tensor]):
        out, new_conv_cache = self.decoder(z, conv_cache=conv_cache)

        # Write all new caches at the end of the graph, as in the Wan VAE. Writing a cache right after its conv
        # corrupts the decoder_n output: the write does not depend on the read of the old cache by the same conv,
        # so the device can run it first.
        position = torch.tensor(0, dtype=torch.int16)
        frame_axis = torch.tensor(2, dtype=torch.int16)
        updates = [
            torch.ops.rbln_custom_ops.rbln_cache_update(
                buffer, cache.to_cache(cache.get(new_conv_cache)).to(buffer.dtype), position, frame_axis
            )
            for cache, buffer in zip(self.static_caches, buffers, strict=True)
        ]
        return out, self.conv_in_cache.get(new_conv_cache).contiguous(), updates


class _VAECogVideoXDecoder0(_VAECogVideoXDecoder):
    """`decoder_0`: the first chunk, which starts without conv caches."""

    def forward(self, z: torch.Tensor, *buffers: torch.Tensor):
        return self._decode(z, None, buffers)


class _VAECogVideoXDecoderN(_VAECogVideoXDecoder):
    """`decoder_n`: a later chunk, which reads the conv caches of the chunk before it."""

    def forward(self, z: torch.Tensor, conv_in_cache: torch.Tensor, *buffers: torch.Tensor):
        conv_cache = {}
        self.conv_in_cache.put(conv_cache, conv_in_cache)
        for cache, buffer in zip(self.static_caches, buffers, strict=True):
            cache.put(conv_cache, cache.from_cache(buffer).to(z.dtype))
        return self._decode(z, conv_cache, buffers)


class _VAECogVideoXEncoder(torch.nn.Module):
    """
    `encoder`: the CogVideoX VAE encoder on a 1-frame image (image-to-video).

    diffusers encodes fewer than `num_sample_frames_batch_size` (8) frames in one call without a conv cache, so
    the image is one graph without static caches.
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

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.image_size = self.rbln_config.image_size
        self.use_slicing = self.rbln_config.use_slicing
        self.use_tiling = False

        # By index: without runtimes, self.model is an UnavailableRuntime (len 0, any index returns itself).
        runtimes = {cfg.compiled_model_name: self.model[i] for i, cfg in enumerate(self.rbln_config.compile_cfgs)}
        # The chunk sizes are the frame axis of the compiled z inputs. Without decoder_n, the graphs decode
        # exactly the first chunk, which a frame batch of the same size expresses.
        chunk = {cfg.compiled_model_name: cfg.input_info[0][1][2] for cfg in self.rbln_config.compile_cfgs}
        self.decoder = RBLNRuntimeCogVideoXVAEDecoder(
            runtime=runtimes["decoder_0"],
            decoder_n=runtimes.get("decoder_n"),
            main_input_name="z",
            first_chunk_size=chunk["decoder_0"],
            frame_batch_size=chunk.get("decoder_n", chunk["decoder_0"]),
            dtype=self.rbln_config.dtype,
        )
        self.encoder = None
        if "encoder" in runtimes:
            self.encoder = RBLNRuntimeCogVideoXVAEEncoder(
                runtime=runtimes["encoder"], main_input_name="x", dtype=self.rbln_config.dtype
            )

    @staticmethod
    def _latent_size(model: AutoencoderKLCogVideoX, rbln_config: RBLNAutoencoderKLCogVideoXConfig) -> tuple[int, int]:
        spatial_compression = 2 ** (len(model.config.block_out_channels) - 1)
        return rbln_config.height // spatial_compression, rbln_config.width // spatial_compression

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNAutoencoderKLCogVideoXConfig
    ) -> tuple[torch.nn.Module, ...]:
        """One graph per compile config, in the same order."""
        latent_height, latent_width = cls._latent_size(model, rbln_config)
        graphs = {
            "decoder_0": lambda: _VAECogVideoXDecoder0(model, latent_height, latent_width),
            "decoder_n": lambda: _VAECogVideoXDecoderN(model, latent_height, latent_width),
            "encoder": lambda: _VAECogVideoXEncoder(model),
        }
        return tuple(graphs[cfg.compiled_model_name]().eval() for cfg in rbln_config.compile_cfgs)

    @classmethod
    def get_compiled_model(
        cls, model, rbln_config: RBLNAutoencoderKLCogVideoXConfig
    ) -> dict[str, rebel.RBLNCompiledModel]:
        cfgs = {cfg.compiled_model_name: cfg for cfg in rbln_config.compile_cfgs}
        graphs = dict(zip(cfgs, cls._wrap_model_if_needed(model, rbln_config), strict=True))

        # decoder_0 and decoder_n wrap the same vae.decoder and share its weights on the device. They also share
        # the static cache buffers: decoder_n gets the same dummy tensors as decoder_0.
        context = CompileContext(use_weight_sharing=True)
        example_inputs = {"decoder_0": cfgs["decoder_0"].get_dummy_inputs(fill=0)}
        buffers = {
            name: tensor
            for (name, _, _), tensor in zip(cfgs["decoder_0"].input_info, example_inputs["decoder_0"], strict=True)
            if name != "z"
        }
        for tensor in buffers.values():
            context.mark_static_address(tensor)
        if "decoder_n" in cfgs:
            example_inputs["decoder_n"] = cfgs["decoder_n"].get_dummy_inputs(fill=0, static_tensors=buffers)
        if "encoder" in cfgs:
            example_inputs["encoder"] = cfgs["encoder"].get_dummy_inputs(fill=0)

        # A graph reuses the intermediate buffer of a graph compiled before it in the same context only if that
        # buffer is large enough. decoder_n needs the largest one (7.25 GiB vs 4.40 GiB for decoder_0 at 480x720,
        # 49 frames), so it goes first; in the other order the graphs do not fit on one device together.
        compiled = {}
        for name in sorted(graphs, key=("decoder_n", "decoder_0", "encoder").index):
            compiled[name] = cls.compile(
                graphs[name],
                rbln_compile_config=cfgs[name],
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device_map[name],
                example_inputs=example_inputs[name],
                compile_context=context,
            )
        return {name: compiled[name] for name in cfgs}

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        transformer_config = pipe.transformer.config
        vae_scale_factor_spatial = pipe.vae_scale_factor_spatial
        if rbln_config.vae.height is None:
            rbln_config.vae.height = transformer_config.sample_height * vae_scale_factor_spatial
        if rbln_config.vae.width is None:
            rbln_config.vae.width = transformer_config.sample_width * vae_scale_factor_spatial
        if rbln_config.vae.num_frames is None:
            rbln_config.vae.num_frames = transformer_config.sample_frames
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

        batch_size = rbln_config.batch_size
        dtype = rbln_config.dtype
        # Same chunks as AutoencoderKLCogVideoX._decode: `num_latent_frames_batch_size` (2) latent frames per call,
        # and the first call also takes the odd remainder frame.
        frame_batch = model.num_latent_frames_batch_size
        num_latent_frames = (rbln_config.num_frames - 1) // model_config.temporal_compression_ratio + 1
        first_chunk = min(num_latent_frames, frame_batch + num_latent_frames % frame_batch)
        latent_height, latent_width = cls._latent_size(model, rbln_config)
        caches = _decoder_conv_caches(model.decoder, latent_height, latent_width)
        buffer_info = [
            (f"conv_cache_{i}", cache.compile_shape(batch_size), dtype) for i, cache in enumerate(caches[1:], start=1)
        ]

        def z_info(num_latent_frames: int):
            return (
                "z",
                [batch_size, model_config.latent_channels, num_latent_frames, latent_height, latent_width],
                dtype,
            )

        compile_cfgs = [
            RBLNCompileConfig(compiled_model_name="decoder_0", input_info=[z_info(first_chunk)] + buffer_info)
        ]
        if num_latent_frames > first_chunk:
            decoder_n_info = [
                z_info(frame_batch),
                ("conv_cache_0", caches[0].shape(batch_size), dtype),
            ]
            compile_cfgs.append(
                RBLNCompileConfig(compiled_model_name="decoder_n", input_info=decoder_n_info + buffer_info)
            )
        if rbln_config.uses_encoder:
            # Image-to-video encodes one image as a 1-frame video.
            image_info = ("x", [batch_size, model_config.in_channels, 1, rbln_config.height, rbln_config.width], dtype)
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name="encoder", input_info=[image_info]))

        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: list[rebel.RBLNCompiledModel],
        rbln_config: RBLNAutoencoderKLCogVideoXConfig,
    ) -> list[rebel.Runtime]:
        # Each graph goes to its own device from device_map (the base class uses one device for all).
        return [
            rebel.Runtime(
                compiled_model,
                tensor_type="pt",
                device=rbln_config.device_map[cfg.compiled_model_name],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for compiled_model, cfg in zip(compiled_models, rbln_config.compile_cfgs, strict=True)
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
