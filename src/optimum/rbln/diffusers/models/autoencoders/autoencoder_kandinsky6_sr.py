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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import rebel
import torch
from diffusers.models.autoencoders.autoencoder_kandinsky6_sr import Kandinsky6SRCausalConv3d, Kandinsky6SRVAE
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from rebel.compile_context import CompileContext
from torch._subclasses.fake_tensor import FakeTensorMode

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ...configurations import RBLNKandinsky6SRVAEConfig
from .vae import RBLNRuntimeKandinsky6SRVAEDecoder, RBLNRuntimeKandinsky6SRVAEEncoder


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


# frame axis of the (n, c, d, h, w) static caches, for rbln_cache_update
_CACHE_FRAME_AXIS = 2
# rbln_cache_update needs device-aligned (128-byte) rows, so the rows of a cache are padded to this many elements.
_CACHE_ROW_ALIGNMENT = 64
# `conv_in` reads the graph input itself, so the frames it carries to the next segment are the last input frames of
# the segment. The host passes them back as an input instead of the device keeping a cache of a few input channels.
_INPUT_CACHE_PATH = ("conv_in", "padding")


class _Kandinsky6SRCausalConv3d(torch.nn.Module):
    """One causal convolution over the carried frames and the segment, caching frames before spatial padding."""

    def __init__(self, module: Kandinsky6SRCausalConv3d):
        super().__init__()
        self.conv = module.conv
        self.conv.padding = (0, module.height_pad, module.width_pad)
        self.time_pad = module.time_pad
        self.time_stride = module.time_stride

    @staticmethod
    def make_cache() -> dict:
        return {"padding": None}

    def forward(self, hidden_states: torch.Tensor, cache: dict) -> torch.Tensor:
        history = cache["padding"]
        if history is None:
            history = hidden_states[:, :, :1].expand(-1, -1, self.time_pad, -1, -1)
        if history.shape[2]:
            hidden_states = torch.cat([history, hidden_states], dim=2)
        output = self.conv(hidden_states)
        cache["padding"] = hidden_states[:, :, output.shape[2] * self.time_stride :].clone()
        return output


def _wrap_causal_convs(module: torch.nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, Kandinsky6SRCausalConv3d):
            setattr(module, name, _Kandinsky6SRCausalConv3d(child))
        else:
            _wrap_causal_convs(child)


def _padding_paths(cache, prefix: tuple = ()) -> Iterator[tuple]:
    """The key paths of every causal-conv `padding` entry of a `make_cache()` dict, in a fixed order."""
    if isinstance(cache, dict):
        for key, value in cache.items():
            if key == "padding":
                yield (*prefix, key)
            else:
                yield from _padding_paths(value, (*prefix, key))
    elif isinstance(cache, list):
        for index, value in enumerate(cache):
            yield from _padding_paths(value, (*prefix, index))


def _get(cache, path: tuple):
    for key in path:
        cache = cache[key]
    return cache


def _set(cache, path: tuple, value) -> None:
    _get(cache, path[:-1])[path[-1]] = value


def _mark_later_segment(cache) -> None:
    """Turn the first-segment flags of the spatial norms off, as their first call does."""
    if isinstance(cache, dict):
        for key, value in cache.items():
            if key == "is_first_segment":
                cache[key] = False
            else:
                _mark_later_segment(value)
    elif isinstance(cache, list):
        for value in cache:
            _mark_later_segment(value)


@dataclass(frozen=True)
class _SegmentCache:
    """
    The padding one causal conv carries to the next segment: the last `shape[2]` input frames of that conv.

    `path` is its key path in the `make_cache()` dict of the encoder or decoder. The frames are stored before
    spatial padding. When the width is unaligned but the channels are aligned, a (n, h*w, d, c) view avoids row
    padding. Other caches keep (n, c, d, h, w) with aligned rows. The next convolution applies its spatial padding.
    """

    path: tuple
    shape: tuple[int, int, int, int, int]

    @property
    def channels_last(self) -> bool:
        return self.shape[1] % _CACHE_ROW_ALIGNMENT == 0 and self.shape[-1] % _CACHE_ROW_ALIGNMENT != 0

    @property
    def compile_shape(self) -> list[int]:
        if self.channels_last:
            n, c, d, h, w = self.shape
            return [n, h * w, d, c]
        *leading, width = self.shape
        return [*leading, -(-width // _CACHE_ROW_ALIGNMENT) * _CACHE_ROW_ALIGNMENT]

    def to_cache(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            return x.permute(0, 3, 4, 2, 1).reshape(self.compile_shape)
        padding = self.compile_shape[-1] - self.shape[-1]
        return torch.nn.functional.pad(x, (0, padding)) if padding else x

    def from_cache(self, cache: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            n, c, d, h, w = self.shape
            return cache.reshape(n, h, w, d, c).permute(0, 4, 3, 1, 2)
        return cache[..., : self.shape[-1]]


def get_segment_caches(
    module: torch.nn.Module, sample_shape: Sequence[int], dtype: torch.dtype
) -> tuple[list[_SegmentCache], list[tuple]]:
    """
    The paddings `module` (a `Kandinsky6SREncoder3D` or `Kandinsky6SRDecoder3D`) carries between segments on the
    device, all but that of `conv_in`.

    Runs one segment on fake tensors to read their shapes. Returns the caches that hold frames, and the paths of the
    paddings that stay empty between segments (the stride-2 temporal convs consume whole frame pairs).
    """
    cache = module.make_cache()
    with FakeTensorMode(allow_non_fake_inputs=True):
        module(torch.empty(*sample_shape, dtype=dtype), cache)
    caches, empty_paths = [], []
    for path in _padding_paths(cache):
        shape = tuple(_get(cache, path).shape)
        if path == _INPUT_CACHE_PATH:
            continue
        if shape[_CACHE_FRAME_AXIS] == 0:
            empty_paths.append(path)
        else:
            caches.append(_SegmentCache(path, shape))
    return caches, empty_paths


def get_history_shape(module: torch.nn.Module, sample_shape: Sequence[int]) -> list[int]:
    """The shape of the last input frames of a segment that `conv_in` of the next segment reads."""
    batch_size, channels, _, height, width = sample_shape
    return [batch_size, channels, module.conv_in.time_pad, height, width]


class _VAEKandinsky6SRSegment(torch.nn.Module):
    """
    One segment of `Kandinsky6SRVAE.encode` or `.decode` as a graph.

    The causal-conv paddings stay on the device as static buffers (graph inputs after the sample), read by the later
    segments and rewritten by every segment with `rbln_cache_update`. The first segment starts from `make_cache()`;
    a later one also takes the last input frames of the segment before it, the padding of `conv_in`.
    """

    def __init__(self, module: torch.nn.Module, caches: list[_SegmentCache], empty_paths: list[tuple], first: bool):
        super().__init__()
        self.module = module
        self.caches = caches
        self.empty_paths = empty_paths
        self.first = first

    def forward(self, sample: torch.Tensor, *inputs: torch.Tensor):
        cache = self.module.make_cache()
        buffers = inputs
        if not self.first:
            history, *buffers = inputs
            _mark_later_segment(cache)
            _set(cache, _INPUT_CACHE_PATH, history)
            for segment_cache, buffer in zip(self.caches, buffers, strict=True):
                _set(cache, segment_cache.path, segment_cache.from_cache(buffer))
            for path in self.empty_paths:
                _set(cache, path, sample.new_zeros(1, 1, 0, 1, 1))
        output = self.module(sample, cache)

        position = torch.tensor(0, dtype=torch.int16)
        frame_axis = torch.tensor(_CACHE_FRAME_AXIS, dtype=torch.int16)
        for segment_cache, buffer in zip(self.caches, buffers, strict=True):
            torch.ops.rbln_custom_ops.rbln_cache_update(
                buffer, segment_cache.to_cache(_get(cache, segment_cache.path)).to(buffer.dtype), position, frame_axis
            )
        return output


class RBLNKandinsky6SRVAE(RBLNModel):
    """
    RBLN implementation of Kandinsky6SRVAE, the causal video K-VAE of the Kandinsky 6 super-resolution pipeline.

    diffusers encodes and decodes a video in temporal segments of 16 pixel frames (plus the leading frame), carrying
    the causal-conv paddings from one segment to the next. Here the paddings stay on the device as static buffers and
    the segment loop runs on the host: `encoder_0` / `decoder_0` take the first segment and `encoder_n` /
    `decoder_n` every later one, of `segment_frames` pixel frames. A shorter last segment is padded, which the causal
    convs keep out of the frames before it, so the graphs serve any `1 + 4k` frame count.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    auto_model_class = Kandinsky6SRVAE
    hf_library_name = "diffusers"
    _rbln_config_class = RBLNKandinsky6SRVAEConfig

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.image_size = self.rbln_config.image_size
        self.use_slicing = self.rbln_config.use_slicing
        self.spatial_compression_ratio = 2 ** (len(self.config.encoder_block_out_channels) - 1)
        self.temporal_compression_ratio = self.config.temporal_compression_ratio

        # By index: without runtimes, self.model is an UnavailableRuntime (len 0, any index returns itself).
        runtimes = {cfg.compiled_model_name: self.model[i] for i, cfg in enumerate(self.rbln_config.compile_cfgs)}
        # The sample and the history inputs of a `_n` graph.
        later_inputs = {
            cfg.compiled_model_name: (cfg.input_info[0][1], cfg.input_info[1][1])
            for cfg in self.rbln_config.compile_cfgs
            if cfg.compiled_model_name.endswith("_n")
        }
        segment_shape, history_shape = later_inputs["decoder_n"]
        self.decoder = RBLNRuntimeKandinsky6SRVAEDecoder(
            runtime=runtimes["decoder_0"],
            decoder_n=runtimes["decoder_n"],
            main_input_name="z",
            segment_length=segment_shape[2],
            history_frames=history_shape[2],
            temporal_compression_ratio=self.temporal_compression_ratio,
            dtype=self.rbln_config.dtype,
        )
        self.encoder = None
        if self.rbln_config.uses_encoder:
            segment_shape, history_shape = later_inputs["encoder_n"]
            self.encoder = RBLNRuntimeKandinsky6SRVAEEncoder(
                runtime=runtimes["encoder_0"],
                encoder_n=runtimes["encoder_n"],
                main_input_name="x",
                segment_length=segment_shape[2],
                history_frames=history_shape[2],
                temporal_compression_ratio=self.temporal_compression_ratio,
                dtype=self.rbln_config.dtype,
            )

    @staticmethod
    def _segment_shapes(model_config, rbln_config: RBLNKandinsky6SRVAEConfig) -> dict[str, list[int]]:
        """The sample shape of every graph: the first segment, which also takes the leading frame, and the later ones."""
        batch_size = rbln_config.batch_size
        segment_frames = rbln_config.segment_frames
        latent_segment = segment_frames // model_config.temporal_compression_ratio
        latent_height, latent_width = rbln_config.latent_sample_size
        shapes = {
            "decoder_0": [batch_size, model_config.latent_channels, latent_segment + 1, latent_height, latent_width],
            "decoder_n": [batch_size, model_config.latent_channels, latent_segment, latent_height, latent_width],
        }
        if rbln_config.uses_encoder:
            height, width = rbln_config.sample_size
            shapes["encoder_0"] = [batch_size, model_config.in_channels, segment_frames + 1, height, width]
            shapes["encoder_n"] = [batch_size, model_config.in_channels, segment_frames, height, width]
        return shapes

    @classmethod
    def _segment_graphs(
        cls, model: Kandinsky6SRVAE, rbln_config: RBLNKandinsky6SRVAEConfig
    ) -> dict[str, _VAEKandinsky6SRSegment]:
        shapes = cls._segment_shapes(model.config, rbln_config)
        graphs = {}
        for name, module in (("encoder", model.encoder), ("decoder", model.decoder)):
            if f"{name}_0" not in shapes:
                continue
            _wrap_causal_convs(module)
            caches, empty_paths = get_segment_caches(module, shapes[f"{name}_0"], model.dtype)
            graphs[f"{name}_0"] = _VAEKandinsky6SRSegment(module, caches, empty_paths, first=True).eval()
            graphs[f"{name}_n"] = _VAEKandinsky6SRSegment(module, caches, empty_paths, first=False).eval()
        return graphs

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNKandinsky6SRVAEConfig
    ) -> tuple[torch.nn.Module, ...]:
        """One graph per compile config, in the same order."""
        graphs = cls._segment_graphs(model, rbln_config)
        return tuple(graphs[cfg.compiled_model_name] for cfg in rbln_config.compile_cfgs)

    @classmethod
    def get_compiled_model(cls, model, rbln_config: RBLNKandinsky6SRVAEConfig) -> dict[str, rebel.RBLNCompiledModel]:
        cfgs = {cfg.compiled_model_name: cfg for cfg in rbln_config.compile_cfgs}
        graphs = dict(zip(cfgs, cls._wrap_model_if_needed(model, rbln_config), strict=True))

        compiled = {}
        for name in ("encoder", "decoder"):
            if f"{name}_0" not in cfgs:
                continue
            # The first and the later segments run the same weights and share the static cache buffers.
            context = CompileContext(use_weight_sharing=True)
            first_inputs = cfgs[f"{name}_0"].get_dummy_inputs(fill=0)
            buffers = {
                cache_name: tensor
                for (cache_name, _, _), tensor in zip(cfgs[f"{name}_0"].input_info, first_inputs, strict=True)
                if cache_name.startswith("cache_")
            }
            for tensor in buffers.values():
                context.mark_static_address(tensor)
            example_inputs = {
                f"{name}_0": first_inputs,
                f"{name}_n": cfgs[f"{name}_n"].get_dummy_inputs(fill=0, static_tensors=buffers),
            }
            for graph_name in (f"{name}_n", f"{name}_0"):
                compiled[graph_name] = cls.compile(
                    graphs[graph_name],
                    rbln_compile_config=cfgs[graph_name],
                    create_runtimes=rbln_config.create_runtimes,
                    device=rbln_config.device_map[graph_name],
                    example_inputs=example_inputs[graph_name],
                    compile_context=context,
                )
        return {name: compiled[name] for name in cfgs}

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        layout = pipe.get_tile_layout(rbln_config)
        if rbln_config.vae.sample_size is None:
            rbln_config.vae.sample_size = layout.video_size
        if rbln_config.vae.latent_sample_size is None:
            rbln_config.vae.latent_sample_size = layout.latent_base_tile_size
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor] | None,
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNKandinsky6SRVAEConfig,
    ) -> RBLNKandinsky6SRVAEConfig:
        required = ["latent_sample_size"] + (["sample_size"] if rbln_config.uses_encoder else [])
        missing = [key for key in required if getattr(rbln_config, key) is None]
        if missing:
            raise ValueError(
                f"{', '.join(f'`{key}`' for key in missing)} must be specified to compile the Kandinsky 6 SR VAE. "
                "Compiling through a pipeline fills them from the input video; a standalone compile must pass them "
                "in rbln_config."
            )

        ratio = model_config.temporal_compression_ratio
        if rbln_config.segment_frames is None:
            rbln_config.segment_frames = ratio
        if rbln_config.segment_frames <= 0 or rbln_config.segment_frames % ratio:
            raise ValueError(
                f"`segment_frames` must be a positive multiple of the temporal compression {ratio}, got "
                f"{rbln_config.segment_frames}."
            )

        dtype = rbln_config.dtype
        shapes = cls._segment_shapes(model_config, rbln_config)
        compile_cfgs = []
        for name, module in (("encoder", model.encoder), ("decoder", model.decoder)):
            if f"{name}_0" not in shapes:
                continue
            _wrap_causal_convs(module)
            caches, _ = get_segment_caches(module, shapes[f"{name}_0"], model.dtype)
            buffer_info = [(f"cache_{i}", cache.compile_shape, dtype) for i, cache in enumerate(caches)]
            sample_name = "x" if name == "encoder" else "z"
            history_info = (f"{sample_name}_history", get_history_shape(module, shapes[f"{name}_n"]), dtype)
            first_info = [(sample_name, shapes[f"{name}_0"], dtype), *buffer_info]
            later_info = [(sample_name, shapes[f"{name}_n"], dtype), history_info, *buffer_info]
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name=f"{name}_0", input_info=first_info))
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name=f"{name}_n", input_info=later_info))

        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: list[rebel.RBLNCompiledModel],
        rbln_config: RBLNKandinsky6SRVAEConfig,
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
        Encode a video into its latent distribution.

        Args:
            x: The video to encode, shaped (batch, channels, num_frames, height, width), with `1 + 4k` frames.
            return_dict:
                Whether to return output as a dictionary. Defaults to True.

        Returns:
            The latent distribution or AutoencoderKLOutput if return_dict=True
        """
        if self.encoder is None:
            raise ValueError("The VAE was compiled without the encoder; compile it with `uses_encoder=True`.")
        if self.use_slicing and x.shape[0] > 1:
            moments = torch.cat([self.encoder.encode(x_slice) for x_slice in x.split(1)])
        else:
            moments = self.encoder.encode(x)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> torch.Tensor | DecoderOutput:
        """
        Decode latents into a video.

        Args:
            z: The latents to decode, already divided by `scaling_factor`, shaped
                (batch, latent_channels, num_latent_frames, height, width).
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
    "RBLNKandinsky6SRVAE",
]
