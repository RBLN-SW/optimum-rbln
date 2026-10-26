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

import math
import weakref
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import rebel
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21KVCache,
    QwenImage21Rope,
    QwenImage21TimestepProjEmbeddings,
    QwenImage21Transformer2DModel,
)
from rebel.compile_context import CompileContext

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....modeling_base import Preprocessor
from ....transformers.modeling_attention_utils import align, set_default_values, validate_attention_method
from ....utils.logging import get_logger
from ...configurations import RBLNQwenImage21Transformer2DModelConfig


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


logger = get_logger(__name__)

# Each vision-language image slot represents a 2x2 group of latent tokens.
_IMG_TOKENS_PER_SLOT = 4
_TEXT_PREFILL = "text_prefill"
_KV_CACHE_PREFIX = "kv_cache_"


def _get_image_chunk_name(num_tokens: int) -> str:
    return f"image_{num_tokens}"


def _get_image_chunk_sizes(rbln_config: RBLNQwenImage21Transformer2DModelConfig) -> list[int]:
    # An image graph depends only on the token count, as the host computes the RoPE tables.
    return sorted({math.prod(size) for size in rbln_config.sample_size})


def _get_kv_cache_len(rbln_config: RBLNQwenImage21Transformer2DModelConfig) -> int:
    """KV-cache slots per prompt.

    A chunk writes its whole size from where it starts. The prefix ends by `max_seq_len`, so its last text chunk
    starts before `max_seq_len` and the target image starts at most there. Flash attention rounds up to whole
    partitions; its partition-alignment padding can still need more, which the chunk plan reports.
    """
    # `_update_rbln_config` sets the chunk size before any cache length is computed.
    num_tokens = max(*_get_image_chunk_sizes(rbln_config), cast(int, rbln_config.prefill_chunk_size))
    # `set_default_values` sets a partition length exactly when the attention is flash.
    partition_len = rbln_config.kvcache_partition_len
    if partition_len is not None:
        return max(align(rbln_config.max_seq_len + num_tokens, partition_len), 2 * partition_len)
    return align(rbln_config.max_seq_len + num_tokens, 64)


def _get_kv_block_size(rbln_config: RBLNQwenImage21Transformer2DModelConfig) -> int:
    # Eager attention keeps one block per prompt; flash attention one block per partition.
    return rbln_config.kvcache_partition_len or _get_kv_cache_len(rbln_config)


class _PagedAttention(torch.nn.Module):
    """Chunk attention over the paged KV cache.

    The op writes the chunk's keys and values into the cache at `seq`, then attends to the valid cache slots in
    `mask` up to the end of the chunk: causally inside the chunk, or bidirectionally with `is_bidirectional`.
    With `partition_len`, flash attention handles the cache one partition (block) at a time.

    One instance serves every graph, so its `scale` is the same parameter in each of them; separate parameters
    stop the compiler from sharing the model weights between the graphs.
    """

    def __init__(self, head_dim: int, partition_len: int | None):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0 / math.sqrt(head_dim)), requires_grad=False)
        self.partition_len = partition_len

    def forward(self, query, key, value, key_cache, value_cache, seq, block_table, mask, is_bidirectional: bool):
        # (1, T, H, D) -> (1, H, 1, T, D)
        query = query.transpose(1, 2).unsqueeze(2)
        key = key.transpose(1, 2).unsqueeze(2)
        value = value.transpose(1, 2).unsqueeze(2)
        cache_args = (key_cache.unsqueeze(2), value_cache.unsqueeze(2), seq, self.scale, block_table)
        block_size = key_cache.shape[2]
        if self.partition_len is None:
            hidden_states = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(
                query, key, value, *cache_args, block_size, is_bidirectional, mask=mask
            )
        else:
            hidden_states = torch.ops.rbln_custom_ops.paged_flash_causal_attn_prefill(
                query, key, value, *cache_args, block_size, self.partition_len, is_bidirectional, mask=mask
            )
        return hidden_states.squeeze(2).transpose(1, 2).flatten(2, 3)


class _QwenImage21ChunkMixin:
    """`QwenImage21TransformerBlock.forward` for one chunk, with the attention done over the KV cache.

    Every block reads the same shared `modulation`, laid out as `[scale1, gate1, scale2, gate2]`.
    """

    def _project_qkv(self, attn, hidden_states, rope_cos, rope_sin):
        head_dim = attn.inner_dim // attn.heads
        query = attn.to_q(hidden_states).unflatten(-1, (-1, head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (-1, head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (-1, head_dim))
        query = attn.norm_q(query).to(value.dtype)
        key = attn.norm_k(key).to(value.dtype)
        # Real form of the upstream complex rotation: `cos`/`sin` repeat each frequency for both channels of a
        # pair. The helper broadcasts them to (1, seq, 1, head_dim), the rank of `x` that the compiler's rotary
        # embedding lowering needs.
        query = apply_rotary_emb(query, (rope_cos, rope_sin), use_real=True, use_real_unbind_dim=-1, sequence_dim=1)
        key = apply_rotary_emb(key, (rope_cos, rope_sin), use_real=True, use_real_unbind_dim=-1, sequence_dim=1)
        return query, key, value

    def _run_blocks(
        self, hidden_states, modulation, rope_cos, rope_sin, seq, block_table, mask, kv_caches, is_bidirectional
    ):
        scale1, gate1, scale2, gate2 = modulation.chunk(4, dim=-1)
        for index, block in enumerate(self.model.transformer_blocks):
            attn = block.attn
            normed = block.img_norm1(hidden_states) * (1 + scale1)
            query, key, value = self._project_qkv(attn, normed, rope_cos, rope_sin)
            attn_output = self.attention(
                query,
                key,
                value,
                kv_caches[2 * index],
                kv_caches[2 * index + 1],
                seq,
                block_table,
                mask,
                is_bidirectional,
            )
            hidden_states = hidden_states + gate1.tanh() * attn.to_out[0](attn_output)

            normed = block.img_norm2(hidden_states) * (1 + scale2)
            hidden_states = hidden_states + gate2.tanh() * block.img_mlp(normed)

            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


class QwenImage21TextChunkWrapper(_QwenImage21ChunkMixin, torch.nn.Module):
    """Causal text chunk: writes the keys and values of `prefill_chunk_size` prompt tokens into the KV cache.

    The prefix modulates from `t = 0` (`causal_condition`), so its keys and values do not depend on the step.
    """

    def __init__(self, model: QwenImage21Transformer2DModel, attention: _PagedAttention) -> None:
        super().__init__()
        self.model = model
        self.attention = attention

    def forward(self, encoder_hidden_states, temb, rope_cos, rope_sin, seq, block_table, mask, *kv_caches):
        hidden_states = self.model.txt_in(encoder_hidden_states)
        modulation = self.model.modulation(temb).unsqueeze(1)
        hidden_states = self._run_blocks(
            hidden_states, modulation, rope_cos, rope_sin, seq, block_table, mask, kv_caches, is_bidirectional=False
        )
        # The chunk's result is the keys and values in the cache; the graph still needs an output.
        return hidden_states[:, :1]


class QwenImage21ImageChunkWrapper(_QwenImage21ChunkMixin, torch.nn.Module):
    """Bidirectional image chunk: one image block attends to the cache and to itself as a whole.

    The target image runs through it at every step with its timestep embedding and gives the noise prediction.
    """

    def __init__(self, model: QwenImage21Transformer2DModel, attention: _PagedAttention) -> None:
        super().__init__()
        self.model = model
        self.attention = attention

    def forward(self, hidden_states, temb, rope_cos, rope_sin, seq, block_table, mask, *kv_caches):
        hidden_states = self.model.img_in(hidden_states)
        modulation = self.model.modulation(temb).unsqueeze(1)
        hidden_states = self._run_blocks(
            hidden_states, modulation, rope_cos, rope_sin, seq, block_table, mask, kv_caches, is_bidirectional=True
        )
        hidden_states = self.model.norm_out(hidden_states, temb)
        return self.model.proj_out(hidden_states)


class _Chunk(NamedTuple):
    image_id: int  # -1 for text, else the image block id
    start: int  # position in the joint sequence: RoPE and inputs
    num_valid: int
    chunk_size: int
    cache_start: int  # KV-cache slot: `start` plus the partition-alignment padding before the chunk


@dataclass
class _KVPrompt:
    """A prompt's share of the KV cache in device memory."""

    owner: "weakref.ref[QwenImage21KVCache] | None"  # the pipeline's `kv_cache`; None after a call without one
    prefix_len: int
    valid: torch.Tensor  # (1, kv_cache_len) bool: the cache slots that hold the prefix
    last_used: int = 0

    def is_free(self) -> bool:
        return self.owner is None or self.owner() is None


class RBLNQwenImage21Transformer2DModel(RBLNModel):
    """
    RBLN implementation of QwenImage21Transformer2DModel, the single-stream transformer of Qwen-Image 2.1.

    The joint text/image sequence runs as chunks over a KV cache that stays in device memory, the way a decoder
    prefills a prompt. The graphs share the model weights and the cache:

    - `text_prefill` runs `prefill_chunk_size` prompt tokens with causal attention.
    - `image_<N>` runs one image block of up to N tokens with bidirectional attention and returns the noise
      prediction. Condition images and the target image both run through it, each in the smallest graph that
      holds it; an image block cannot be split into chunks, as its tokens attend to each other.

    Chunks are tight-packed: each one starts right after the valid tokens of the previous one and overwrites its
    padded tail. With flash attention (`kvcache_partition_len`) a chunk never crosses a partition: one that would
    starts at the next partition, and the slots it skips stay empty. An attention mask over the cache slots hides
    every slot that holds no valid token, and RoPE follows the position in the joint sequence, not the cache slot.

    A call with `kv_cache_mode="cached"` runs the target image only. Other calls also prefill the prompt, which
    gives the same result as the upstream single pass over the joint sequence.

    The KV cache holds `kvcache_num_prompts` prompts. The pipeline's `kv_cache` object identifies the cache of its
    prompt, so classifier-free guidance keeps the positive and the negative prompt side by side.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = QwenImage21Transformer2DModel
    _rbln_config_class = RBLNQwenImage21Transformer2DModelConfig
    _output_class = Transformer2DModelOutput

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)

        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.time_text_embed = QwenImage21TimestepProjEmbeddings(embedding_dim=inner_dim)
        self.time_text_embed.load_state_dict(artifacts["time_text_embed"])
        self.time_text_embed.to(self.rbln_config.dtype)
        self.pos_embed = QwenImage21Rope(theta=10000, axes_dim=list(self.config.axes_dims_rope))

        # Indexed, not zipped: without runtimes, `self.model` is one `UnavailableRuntime` for every index.
        runtimes = {cfg.compiled_model_name: self.model[i] for i, cfg in enumerate(self.rbln_config.compile_cfgs)}
        self.text_prefill = runtimes[_TEXT_PREFILL]
        self.image_chunks = {
            num_tokens: runtimes[_get_image_chunk_name(num_tokens)]
            for num_tokens in _get_image_chunk_sizes(self.rbln_config)
        }
        self.kv_cache_len = _get_kv_cache_len(self.rbln_config)
        self.kv_blocks_per_prompt = self.kv_cache_len // _get_kv_block_size(self.rbln_config)
        self._kv_prompts: list[_KVPrompt | None] = [None] * self.rbln_config.kvcache_num_prompts
        self._kv_clock = 0

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNQwenImage21Transformer2DModelConfig,
    ):
        save_dict = {"time_text_embed": model.time_text_embed.state_dict()}
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNQwenImage21Transformer2DModelConfig
    ) -> dict[str, torch.nn.Module]:
        attention = _PagedAttention(model.config.attention_head_dim, rbln_config.kvcache_partition_len)
        image_chunk = QwenImage21ImageChunkWrapper(model, attention).eval()
        wrapped_models = {_TEXT_PREFILL: QwenImage21TextChunkWrapper(model, attention).eval()}
        for num_tokens in _get_image_chunk_sizes(rbln_config):
            wrapped_models[_get_image_chunk_name(num_tokens)] = image_chunk
        return wrapped_models

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(
        cls, model: "PreTrainedModel", rbln_config: RBLNQwenImage21Transformer2DModelConfig
    ) -> dict[str, rebel.RBLNCompiledModel]:
        wrapped_models = cls._wrap_model_if_needed(model, rbln_config)
        # Every graph wraps the same model and reads the same KV cache: share the weights, and give the cache
        # one static address. Meta tensors keep the cache out of host memory during compilation.
        context = CompileContext(use_weight_sharing=True)
        static_tensors = {}
        compiled_models = {}
        for compile_config in rbln_config.compile_cfgs:
            cache_names = [name for name, _, _ in compile_config.input_info if name.startswith(_KV_CACHE_PREFIX)]
            example_inputs = compile_config.get_dummy_inputs(
                fill=0, static_tensors=static_tensors, meta_tensor_names=cache_names
            )
            for (name, _, _), tensor in zip(compile_config.input_info, example_inputs, strict=True):
                if name in cache_names and name not in static_tensors:
                    static_tensors[name] = tensor
                    context.mark_static_address(tensor, name)

            name = compile_config.compiled_model_name
            compiled_models[name] = cls.compile(
                wrapped_models[name],
                compile_config,
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device_map[name],
                example_inputs=example_inputs,
                compile_context=context,
            )
        return compiled_models

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        # The pipeline config already sets the image sizes and sequence lengths.
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor] | None,
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNQwenImage21Transformer2DModelConfig,
    ) -> RBLNQwenImage21Transformer2DModelConfig:
        if rbln_config.batch_size != 1:
            raise NotImplementedError("RBLNQwenImage21Transformer2DModel supports batch_size=1 only.")
        if not model_config.causal_condition:
            # The prefix keys and values are computed once and reused at every step, which needs a step-independent
            # prefix.
            raise NotImplementedError("RBLNQwenImage21Transformer2DModel supports `causal_condition=True` only.")
        if rbln_config.sample_size is None:
            # 1024x1024, the pipeline default, at 16 pixels per latent token.
            rbln_config.sample_size = [(64, 64)]
        if rbln_config.max_seq_len is None:
            rbln_config.max_seq_len = 512
        if rbln_config.kvcache_num_prompts is None:
            rbln_config.kvcache_num_prompts = 1
        rbln_config.attn_impl, rbln_config.kvcache_partition_len, _, rbln_config.prefill_chunk_size = (
            set_default_values(
                attn_impl=rbln_config.attn_impl,
                kvcache_partition_len=rbln_config.kvcache_partition_len,
                max_seq_len=rbln_config.max_seq_len,
                prefill_chunk_size=rbln_config.prefill_chunk_size,
                npu=rbln_config.npu,
            )
        )
        kv_cache_len = _get_kv_cache_len(rbln_config)
        kv_block_size = _get_kv_block_size(rbln_config)
        # The KV cache's dynamic axis: the per-prompt cache for eager, one partition for flash attention.
        validate_attention_method(
            rbln_config.attn_impl, rbln_config.kvcache_partition_len, kv_block_size, kv_cache_len, rbln_config.npu
        )
        if rbln_config.attn_impl == "flash_attn":
            too_large = [n for n in _get_image_chunk_sizes(rbln_config) if n > rbln_config.kvcache_partition_len]
            if too_large:
                raise ValueError(
                    f"Image blocks of {too_large} tokens do not fit one KV-cache partition "
                    f"(kvcache_partition_len={rbln_config.kvcache_partition_len}): an image block runs as one "
                    "chunk, and a chunk cannot cross a partition. Increase kvcache_partition_len."
                )

        num_layers = model_config.num_layers
        num_heads = model_config.num_attention_heads
        head_dim = model_config.attention_head_dim
        inner_dim = num_heads * head_dim
        dtype = rbln_config.dtype
        chunk_size = rbln_config.prefill_chunk_size
        blocks_per_prompt = kv_cache_len // kv_block_size

        # (num_blocks, heads, block_size, head_dim); a prompt owns `blocks_per_prompt` consecutive blocks.
        num_blocks = rbln_config.kvcache_num_prompts * blocks_per_prompt
        kv_cache_info = [
            (f"{_KV_CACHE_PREFIX}{index}", [num_blocks, num_heads, kv_block_size, head_dim], dtype)
            for index in range(2 * num_layers)
        ]

        def chunk_input_info(input_name: str, num_tokens: int, width: int) -> list:
            # The order of the chunk wrappers' `forward` arguments.
            return [
                (input_name, [1, num_tokens, width], dtype),
                ("temb", [1, inner_dim], dtype),
                ("rope_cos", [num_tokens, head_dim], "float32"),
                ("rope_sin", [num_tokens, head_dim], "float32"),
                ("seq", [1, 1], "int32"),
                ("block_table", [blocks_per_prompt], "int16"),
                ("mask", [1, kv_cache_len], dtype),
                *kv_cache_info,
            ]

        compile_cfgs = [
            RBLNCompileConfig(
                compiled_model_name=_TEXT_PREFILL,
                input_info=chunk_input_info("encoder_hidden_states", chunk_size, model_config.context_in_dim),
            )
        ]
        compile_cfgs += [
            RBLNCompileConfig(
                compiled_model_name=_get_image_chunk_name(num_tokens),
                input_info=chunk_input_info("hidden_states", num_tokens, model_config.in_channels),
            )
            for num_tokens in _get_image_chunk_sizes(rbln_config)
        ]
        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: list[rebel.RBLNCompiledModel],
        rbln_config: RBLNQwenImage21Transformer2DModelConfig,
    ) -> list[rebel.Runtime]:
        names = [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs]
        if any(name not in rbln_config.device_map for name in names):
            cls._raise_missing_compiled_file_error(names)

        return [
            rebel.Runtime(
                compiled_model,
                tensor_type="pt",
                device=rbln_config.device_map[name],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for name, compiled_model in zip(names, compiled_models, strict=True)
        ]

    @contextmanager
    def cache_context(self, name: str):
        # QwenImage21Pipeline wraps each transformer call in diffusers' CacheMixin.cache_context.
        # The compiled graphs carry no diffusers cache hooks, so there is nothing to set.
        yield

    @property
    def transformer_blocks(self) -> range:
        # QwenImage21Pipeline sizes its KV cache by `len(transformer.transformer_blocks)`.
        return range(self.config.num_layers)

    @staticmethod
    def _get_rope_cos_sin(rotary_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # One complex frequency per channel pair -> one real value per channel.
        cos = torch.stack([rotary_emb.real, rotary_emb.real], dim=-1).flatten(-2)
        sin = torch.stack([rotary_emb.imag, rotary_emb.imag], dim=-1).flatten(-2)
        return cos, sin

    def _resolve_image_chunk(self, num_tokens: int) -> int:
        """The smallest image graph that holds an image block of `num_tokens` latent tokens.

        An image block runs as one bidirectional chunk and cannot be split, so it must fit one graph; the graph's
        remaining tokens are padding that the attention mask hides.
        """
        fits = [size for size in self.image_chunks if size >= num_tokens]
        if not fits:
            raise ValueError(
                f"An image block has {num_tokens} latent tokens, more than the largest image graph "
                f"({max(self.image_chunks)} tokens) holds. An image block runs as one chunk, so add a latent "
                f"(height, width) of at least {num_tokens} tokens to sample_size."
            )
        return min(fits)

    def _plan_chunks(self, image_ids: torch.Tensor, prefix_len: int, num_target_tokens: int) -> list[_Chunk]:
        """Tight-packed chunks over the joint sequence: the prefix, then the target image.

        Text runs split into causal chunks of `prefill_chunk_size`; each image block is one bidirectional chunk in
        the smallest image graph that holds it. Image blocks follow the upstream block ids, so two adjacent condition
        images stay separate. A chunk starts right after the valid tokens of the previous one and overwrites the
        padded tail it wrote. With flash attention, a chunk that would cross a partition starts at the next
        partition instead, as the decoder-only runtime does; the slots in between hold no valid token.
        """
        ids = image_ids.tolist()
        runs = []
        start = 0
        while start < prefix_len:
            image_id = ids[start]
            end = start
            if image_id >= 0:
                while end < prefix_len and ids[end] == image_id:
                    end += 1
                chunk_size = self._resolve_image_chunk(end - start)
            else:
                chunk_size = self.rbln_config.prefill_chunk_size
                while end < prefix_len and ids[end] < 0 and end - start < chunk_size:
                    end += 1
            runs.append((image_id, start, end - start, chunk_size))
            start = end
        runs.append((ids[prefix_len], prefix_len, num_target_tokens, self._resolve_image_chunk(num_target_tokens)))

        partition_len = self.rbln_config.kvcache_partition_len
        chunks = []
        padded = 0
        for image_id, start, num_valid, chunk_size in runs:
            cache_start = start + padded
            if partition_len is not None:
                offset_in_partition = cache_start % partition_len
                if offset_in_partition + chunk_size > partition_len:
                    padded += partition_len - offset_in_partition
                    cache_start = start + padded
            if cache_start + chunk_size > self.kv_cache_len:
                raise ValueError(
                    f"The chunk at position {start} needs KV-cache slots up to {cache_start + chunk_size}, past the "
                    f"cache length ({self.kv_cache_len}); partition-alignment padding takes {padded} of them. "
                    "Increase max_seq_len."
                )
            chunks.append(_Chunk(image_id, start, num_valid, chunk_size, cache_start))
        return chunks

    def _find_kv_prompt(self, kv_cache: QwenImage21KVCache) -> int | None:
        for index, prompt in enumerate(self._kv_prompts):
            if prompt is not None and prompt.owner is not None and prompt.owner() is kv_cache:
                return index
        return None

    def _claim_kv_prompt(self, kv_cache: QwenImage21KVCache | None) -> int:
        """Picks the prompt cache a prefill writes: the one `kv_cache` already holds, else a free one, else the
        least recently used one."""
        index = self._find_kv_prompt(kv_cache) if kv_cache is not None else None
        if index is not None:
            return index
        free = [i for i, prompt in enumerate(self._kv_prompts) if prompt is None or prompt.is_free()]
        if free:
            return free[0]
        return min((prompt.last_used, i) for i, prompt in enumerate(self._kv_prompts) if prompt is not None)[1]

    def _run_chunk(self, runtime, inputs, temb, rope_cos, rope_sin, chunk: _Chunk, block_table, valid):
        """Runs one chunk at its cache slot, padded to the graph's chunk size. `rope_cos`/`rope_sin` cover the
        chunk's valid tokens."""
        pad = chunk.chunk_size - chunk.num_valid
        return runtime(
            F.pad(inputs, (0, 0, 0, pad)).contiguous(),
            temb.contiguous(),
            F.pad(rope_cos, (0, 0, 0, pad), value=1.0).contiguous(),
            F.pad(rope_sin, (0, 0, 0, pad)).contiguous(),
            torch.tensor([[chunk.cache_start]], dtype=torch.int32),
            block_table,
            valid.to(self.rbln_config.dtype).contiguous(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        kv_cache: QwenImage21KVCache | None = None,
        kv_cache_mode: str | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple:
        """
        Forward pass for the RBLN-optimized QwenImage21Transformer2DModel.

        Args:
            hidden_states (torch.Tensor): Packed latents, condition images first and the target image last, shaped
                (batch_size, image_sequence_length, in_channels).
            encoder_hidden_states (torch.Tensor): Text embeddings from the vision-language encoder.
            timestep (torch.Tensor): Current denoising step, scaled to [0, 1].
            img_shapes (list[list[tuple[int, int, int]]]): Per-sample (frame, height, width) of the images, in latent
                tokens, condition images first and the target image last.
            img_mask (torch.Tensor): True at the image slots of the joint sequence.
            encoder_hidden_states_mask (torch.Tensor | None): Marks the valid text tokens.
            attention_kwargs (dict[str, Any] | None): Ignored.
            kv_cache (QwenImage21KVCache | None): Identifies the prompt whose keys and values the KV cache holds.
                The keys and values stay in device memory.
            kv_cache_mode (str | None): `"extract"` prefills the KV cache, `"cached"` reuses it.
            return_dict (bool): Whether to return a `Transformer2DModelOutput` instead of a plain tuple.

        Returns:
            (`~diffusers.models.modeling_outputs.Transformer2DModelOutput` | tuple): The prediction for the target
            image tokens.
        """
        if kv_cache is not None and kv_cache_mode not in ("extract", "cached"):
            raise ValueError(
                f"kv_cache_mode must be 'extract' or 'cached' when kv_cache is provided, got {kv_cache_mode!r}."
            )
        if kv_cache is None and kv_cache_mode is not None:
            raise ValueError(f"kv_cache_mode is {kv_cache_mode!r} but no kv_cache was passed to hold the prefix.")

        batch_size = hidden_states.shape[0]
        if batch_size != self.rbln_config.batch_size:
            raise ValueError(
                f"Mismatch between the transformer's runtime batch size ({batch_size}) and compiled batch size "
                f"({self.rbln_config.batch_size})."
            )
        image_lengths = [math.prod(shape) for shape in img_shapes[0]]
        if hidden_states.shape[1] != sum(image_lengths):
            raise ValueError(
                f"img_shapes accounts for {sum(image_lengths)} latent tokens but hidden_states has "
                f"{hidden_states.shape[1]}."
            )
        num_tokens = image_lengths[-1]

        # Joint-sequence layout, as upstream builds it. Each image slot of `img_mask` stands for 2x2 latent tokens:
        # `vlm_index` maps every joint position to its vision-language position.
        vlm_index = torch.arange(img_mask.shape[1]).repeat_interleave(
            torch.where(img_mask[0], _IMG_TOKENS_PER_SLOT, 1)
        )
        image_pad_mask = img_mask[0][vlm_index]
        image_ids, target_token_mask = QwenImage21Transformer2DModel.build_token_metadata(
            image_pad_mask, img_shapes[0]
        )
        prefix_len = int((~target_token_mask).sum())
        if prefix_len > self.rbln_config.max_seq_len:
            raise ValueError(
                f"The prompt and the condition images take {prefix_len} tokens, but the transformer was compiled for "
                f"max_seq_len={self.rbln_config.max_seq_len}."
            )
        chunks = self._plan_chunks(image_ids, prefix_len, num_tokens)
        rotary_emb = self.pos_embed(img_shapes[0], image_pad_mask, device=image_pad_mask.device)

        dtype = self.rbln_config.dtype
        timestep = timestep.to(dtype)
        # Extra t=0 row: the prefix modulates from it. The second argument only sets the output dtype, as the target
        # hidden states do upstream.
        timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
        temb = self.time_text_embed(timestep, timestep)

        if kv_cache_mode == "cached":
            kv_prompt = self._find_kv_prompt(kv_cache)
            prompt = self._kv_prompts[kv_prompt] if kv_prompt is not None else None
            if kv_prompt is None or prompt is None or prompt.prefix_len != prefix_len:
                raise RuntimeError(
                    "The KV cache in device memory holds no keys and values for this kv_cache: another prompt "
                    "overwrote them. Each prompt needs its own cache: compile with kvcache_num_prompts=2 for "
                    "classifier-free guidance."
                )
            block_table = self._get_block_table(kv_prompt)
        else:
            kv_prompt = self._claim_kv_prompt(kv_cache)
            block_table = self._get_block_table(kv_prompt)
            valid = self._prefill(
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                vlm_index,
                image_lengths,
                chunks[:-1],
                temb[-1:],
                rotary_emb[:prefix_len],
                block_table,
            )
            owner = weakref.ref(kv_cache) if kv_cache is not None else None
            prompt = self._kv_prompts[kv_prompt] = _KVPrompt(owner, prefix_len, valid)
        self._kv_clock += 1
        prompt.last_used = self._kv_clock

        # The target block starts right after the prefix (or at the next partition) and overwrites the padded tail
        # of the last prefix chunk.
        target = chunks[-1]
        valid = prompt.valid.clone()
        valid[:, target.cache_start : target.cache_start + num_tokens] = True
        target_cos, target_sin = self._get_rope_cos_sin(rotary_emb[prefix_len:])
        output = self._run_chunk(
            self.image_chunks[target.chunk_size],
            hidden_states[:, -num_tokens:].to(dtype),
            temb[:-1],
            target_cos,
            target_sin,
            target,
            block_table,
            valid,
        )[:, :num_tokens]

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def _get_block_table(self, kv_prompt: int) -> torch.Tensor:
        # A prompt owns `kv_blocks_per_prompt` consecutive blocks of the cache.
        first = kv_prompt * self.kv_blocks_per_prompt
        return torch.arange(first, first + self.kv_blocks_per_prompt, dtype=torch.int16)

    def _prefill(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        vlm_index,
        image_lengths,
        chunks,
        temb,
        rotary_emb,
        block_table,
    ):
        """Runs the prefix chunks (prompt text and condition images) into the prompt's KV cache and returns its
        valid cache slots, shaped (1, kv_cache_len). `rotary_emb` covers the prefix."""
        rope_cos, rope_sin = self._get_rope_cos_sin(rotary_emb)
        # Right-padded prompt positions are never attended to, as upstream excludes them as keys.
        prefix_len = rotary_emb.shape[0]
        text_valid = torch.ones(prefix_len, dtype=torch.bool)
        if encoder_hidden_states_mask is not None:
            text_valid = encoder_hidden_states_mask[0].bool()[vlm_index[:prefix_len]]
        # Where each image block starts in `hidden_states`.
        image_offsets = [sum(image_lengths[:index]) for index in range(len(image_lengths))]

        valid = torch.zeros(1, self.kv_cache_len, dtype=torch.bool)
        dtype = self.rbln_config.dtype
        for chunk in chunks:
            end = chunk.start + chunk.num_valid
            slots = slice(chunk.cache_start, chunk.cache_start + chunk.num_valid)
            if chunk.image_id < 0:
                valid[0, slots] = text_valid[chunk.start : end]
                inputs = encoder_hidden_states[:, vlm_index[chunk.start : end]]
                runtime = self.text_prefill
            else:
                # A condition image modulates from t = 0 like the text; its noise prediction is not needed.
                valid[0, slots] = True
                offset = image_offsets[chunk.image_id]
                inputs = hidden_states[:, offset : offset + chunk.num_valid]
                runtime = self.image_chunks[chunk.chunk_size]
            self._run_chunk(
                runtime,
                inputs.to(dtype),
                temb,
                rope_cos[chunk.start : end],
                rope_sin[chunk.start : end],
                chunk,
                block_table,
                valid,
            )
        return valid


__all__ = [
    "RBLNQwenImage21Transformer2DModel",
]
