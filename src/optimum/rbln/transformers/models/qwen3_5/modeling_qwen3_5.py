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

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import torch
from rebel.compile_context import CompileContext
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, PretrainedConfig, PreTrainedModel
from transformers.initialization import no_init_weights
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Model,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionModel,
    Qwen3_5VisionPatchEmbed,
    Qwen3_5VisionRotaryEmbedding,
)

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils import logging
from ...cache_utils import FullAttentionKVCacheMeta, LinearAttentionCacheMeta
from ...modeling_outputs import RBLNDecoderOnlyOutput, _validate_output_hidden_states
from ..decoderonly.decoderonly_runtime_utils import RBLNPageTableManager
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyModel, RBLNDecoderOnlyModelForCausalLM
from .configuration_qwen3_5 import (
    RBLNQwen3_5ForConditionalGenerationConfig,  # noqa: F401
    RBLNQwen3_5ModelConfig,  # noqa: F401
    RBLNQwen3_5VisionModelConfig,  # noqa: F401
)
from .qwen3_5_architecture import (
    Qwen3_5_CausalLMWrapper,
    Qwen3_5_LanguageModelWrapper,
    Qwen3_5VisionModelWrapper,
)
from .qwen3_5_runtime_utils import RBLNQwen3_5RuntimeModel


logger = logging.get_logger(__name__)


def _qwen3_5_build_compile_context(compile_config, example_inputs):
    def is_static_state(name: str) -> bool:
        if "past_key_values" in name:
            return True
        return ("conv_state" in name or "recurrent_state" in name) and not name.endswith("_mask")

    context = CompileContext(use_weight_sharing=True)
    static_tensors = {}
    for (name, _, _), tensor in zip(compile_config.input_info, example_inputs, strict=False):
        if not is_static_state(name):
            continue
        static_tensors[name] = tensor
        context.mark_static_address(tensor, name)
    return context, static_tensors


def _qwen3_5_linear_state_shapes(text_config, batch_size: int):
    conv_dim = 2 * (text_config.linear_num_key_heads * text_config.linear_key_head_dim) + (
        text_config.linear_num_value_heads * text_config.linear_value_head_dim
    )
    conv_state_shape = (batch_size, text_config.linear_conv_kernel_dim - 1, conv_dim)
    # recurrent state/mask are 3D (B, Hv*Dk, Dv) — see the get_input_info comment: merging Hv into dim1 keeps
    # the shared static cache laid out identically in the prefill/decode graphs (no channel-pad mismatch).
    recurrent_state_shape = (
        batch_size,
        text_config.linear_num_value_heads * text_config.linear_key_head_dim,
        text_config.linear_value_head_dim,
    )
    return conv_state_shape, recurrent_state_shape


def _qwen3_5_linear_layer_indices(model_config) -> list[int]:
    text_config = model_config.get_text_config()
    if getattr(text_config, "layer_types", None) is None:
        raise ValueError("Qwen3.5 requires `layer_types` in the model config.")
    return [i for i, t in enumerate(text_config.layer_types) if t == "linear_attention"]


def _qwen3_5_setup_hybrid_runtime(model):
    rbln_config = model.rbln_config
    text_config = model.config.get_text_config()
    page_table_manager = RBLNPageTableManager(rbln_config)
    dec_attn_mask = torch.zeros(rbln_config.batch_size, 1, 1, rbln_config.max_seq_len, dtype=model.dtype)

    common_kwargs = {
        "main_input_name": "inputs_embeds" if rbln_config.use_inputs_embeds else "input_ids",
        "embed_tokens": model.embed_tokens,
        "dec_attn_mask": dec_attn_mask,
        "page_table_manager": page_table_manager,
        "rbln_config": rbln_config,
        "config": text_config,
        "state_dtype": model.dtype,
    }

    # Prefill runs one item at a time (batch=1) and writes its `batch_position` slot, so its state MASKS are
    # batch=1 sized (they gate the single slot the graph reads); the underlying cache is max-batch (get_input_info).
    conv_shape, recur_shape = _qwen3_5_linear_state_shapes(text_config, 1)
    model.prefill_decoder = RBLNQwen3_5RuntimeModel(
        runtime=model.model[0],
        phase="prefill",
        batch_size=rbln_config.batch_size,
        logits_last_dim=model.logits_last_dim,
        conv_state_shape=conv_shape,
        recurrent_state_shape=recur_shape,
        **common_kwargs,
    )

    if model.can_generate():
        model.decoders = {}
        for i, batch_size in enumerate(rbln_config.decoder_batch_sizes):
            conv_shape, recur_shape = _qwen3_5_linear_state_shapes(text_config, batch_size)
            model.decoders[batch_size] = RBLNQwen3_5RuntimeModel(
                runtime=model.model[i + 1],
                phase="decode",
                batch_size=batch_size,
                conv_state_shape=conv_shape,
                recurrent_state_shape=recur_shape,
                **common_kwargs,
            )
        model.decoder = model.decoders[rbln_config.batch_size]


class RBLNQwen3_5TextModel(RBLNDecoderOnlyModel):
    """The bare Qwen3.5 text backbone (no LM head).

    Qwen3.5 is a hybrid decoder: `full_attention` layers use the standard paged KV cache, while
    `linear_attention` (GatedDeltaNet) layers carry a `conv_state` + `recurrent_state` instead. The two
    state tensors reuse the layer's two `past_key_values` slots positionally. This class owns the hybrid
    wiring — `get_input_info` (per-layer tensor specs), `setup_runtime` (the mask-injecting
    `RBLNQwen3_5RuntimeModel`), `_get_compile_context` (mark conv/recurrent static) and `_update_rbln_config`
    (validate `layer_types`). `RBLNQwen3_5ForCausalLM` adds the LM head on top, mirroring how
    `RBLNDecoderOnlyModelForCausalLM` extends `RBLNDecoderOnlyModel`.
    """

    _decoder_wrapper_cls = Qwen3_5_CausalLMWrapper
    _use_rotary_emb = True

    def setup_runtime(self):
        _qwen3_5_setup_hybrid_runtime(self)

    @classmethod
    def _get_compile_context(cls, compile_config, example_inputs):
        return _qwen3_5_build_compile_context(compile_config, example_inputs)

    @classmethod
    def _update_rbln_config(cls, preprocessors=None, model=None, model_config=None, rbln_config=None):
        rbln_config.linear_attention_layers = _qwen3_5_linear_layer_indices(model_config)
        rbln_config = super()._update_rbln_config(
            preprocessors=preprocessors, model=model, model_config=model_config, rbln_config=rbln_config
        )
        if rbln_config.gdn_chunk_size is None:
            rbln_config.gdn_chunk_size = rbln_config.prefill_chunk_size
        if rbln_config.gdn_chunk_size > 128:
            raise ValueError(
                f"gdn_chunk_size must be <= 128, got {rbln_config.gdn_chunk_size}. "
                "Larger GatedDeltaNet sub-chunk sizes are not supported yet — "
                "set gdn_chunk_size to a value <= 128 that divides prefill_chunk_size."
            )
        return rbln_config

    @classmethod
    def get_input_info(cls, batch_size, query_length, rbln_config, model_config: PretrainedConfig):
        text_config = model_config.get_text_config()
        if rbln_config.use_position_ids:
            # Qwen3.5 needs no explicit position_ids: the VL path precomputes the mRoPE cos/sin on the
            # host and feeds them as position_emb, and the Text Model path derives contiguous positions from
            # cache_position (mRoPE degenerates to standard RoPE for text-only).
            raise NotImplementedError("use_position_ids is not supported for the Qwen3.5 model.")
        num_attention_heads = getattr(text_config, "n_head", None) or text_config.num_attention_heads
        num_key_value_heads = getattr(text_config, "num_key_value_heads", None) or num_attention_heads
        num_hidden_layers = getattr(text_config, "n_layer", None) or text_config.num_hidden_layers
        hidden_size = getattr(text_config, "n_embd", None) or text_config.hidden_size
        head_dim = getattr(text_config, "head_dim", None) or hidden_size // num_attention_heads
        is_prefill = query_length > 1

        input_info = []
        if rbln_config.use_inputs_embeds:
            input_info.append(("inputs_embeds", [batch_size, query_length, hidden_size], rbln_config.dtype))
        else:
            input_info.append(("input_ids", [batch_size, query_length], "int64"))

        input_info.append(("cache_position", [batch_size, query_length], "int32"))

        if rbln_config.use_global_attention:
            max_block_cnt = rbln_config.max_seq_len // rbln_config.kvcache_block_size
            input_info.append(
                ("block_tables", [max_block_cnt] if is_prefill else [batch_size, max_block_cnt], "int16")
            )
        if rbln_config.use_local_attention:
            input_info.append(("local_block_tables", [1] if is_prefill else [batch_size, 1], "int16"))

        if cls.use_query_position(rbln_config.use_local_attention, is_prefill, rbln_config.logits_to_keep):
            input_info.append(("query_position", [], "int16"))

        if rbln_config.use_attention_mask:
            input_info.append(
                ("attention_mask", [batch_size, 1, query_length, rbln_config.max_seq_len], rbln_config.dtype)
            )
        if rbln_config.use_lora:
            input_info.append(("lora_int_ids", [batch_size], "int32"))

        # per-layer state: full_attention -> paged KV (key, value); linear_attention -> (conv_state, recurrent_state).
        linear_layers = rbln_config.linear_attention_layers

        if len(rbln_config.cache_metas) > 0:
            input_info.extend([(meta.name, meta.compile_shape, meta.dtype) for meta in rbln_config.cache_metas])
        else:
            kvcache_dtype = rbln_config.dtype
            if rbln_config.quantization and rbln_config.quantization.kv_caches == "fp8":
                kvcache_dtype = "float8_e4m3fn"

            # conv/recurrent caches are one shared static tensor, so sized to the max batch (not batch_size=1 for
            # prefill). Prefill writes its own slot (batch_idx); decode runs the full batch.
            _state_dtype = RBLNCompileConfig.normalize_dtype(rbln_config.dtype)
            cache_metas = []
            for layer_idx in range(num_hidden_layers):
                if layer_idx in linear_layers:
                    # recurrent cache is stored 3D (B, Hv*Dk, Dv); GatedDeltaNet reshapes to 4D internally.
                    conv_shape, recurrent_shape = _qwen3_5_linear_state_shapes(text_config, rbln_config.batch_size)
                    cache_metas.append(
                        LinearAttentionCacheMeta.from_config(
                            f"conv_state_{layer_idx}", layer_idx, shape=list(conv_shape), dtype=_state_dtype
                        )
                    )
                    cache_metas.append(
                        LinearAttentionCacheMeta.from_config(
                            f"recurrent_state_{layer_idx}", layer_idx, shape=list(recurrent_shape), dtype=_state_dtype
                        )
                    )
                else:
                    for slot in range(2):
                        name = f"past_key_values_{layer_idx * 2 + slot}"
                        cache_metas.append(
                            FullAttentionKVCacheMeta.from_config(
                                name,
                                layer_idx,
                                num_key_value_heads,
                                head_dim,
                                RBLNCompileConfig.normalize_dtype(kvcache_dtype),
                                rbln_config,
                            )
                        )
            input_info.extend([(meta.name, meta.compile_shape, meta.dtype) for meta in cache_metas])
            rbln_config.cache_metas.extend(cache_metas)

        # shared 0/1 masks: runtime feeds zeros on prefill window 0 (reset linear state), ones after (carry). See docs.
        if linear_layers:
            # masks match the per-call graph batch (batch_size), so reuse the helper with the same shapes.
            conv_mask_shape, recurrent_mask_shape = _qwen3_5_linear_state_shapes(text_config, batch_size)
            input_info.append(("conv_state_mask", list(conv_mask_shape), rbln_config.dtype))
            input_info.append(("recurrent_state_mask", list(recurrent_mask_shape), rbln_config.dtype))
            # per-token validity (1=real, 0=right-padding); host-built, GatedDeltaNet uses it to drop padding.
            input_info.append(("valid_mask", [batch_size, query_length, 1], rbln_config.dtype))
            # prefill only: which max-batch slot this per-item (batch=1) call reads/writes in the linear caches.
            if is_prefill:
                input_info.append(("batch_idx", [], "int16"))

        return input_info


class RBLNQwen3_5ForCausalLM(RBLNQwen3_5TextModel, RBLNDecoderOnlyModelForCausalLM):
    """
    RBLNQwen3_5ForCausalLM is the text-only (causal language modeling) variant of Qwen3.5, optimized for RBLN NPUs.
    It runs the hybrid Qwen3.5 decoder — GatedDeltaNet `linear_attention` layers interleaved with gated
    `full_attention` layers — without the vision encoder.

    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    Important Note:
        This model includes a Large Language Model (LLM). For optimal performance, it is highly recommended to use
        tensor parallelism for the language model. This can be achieved by using the `rbln_config` parameter in the
        `from_pretrained` method. Refer to the `from_pretrained` documentation and the RBLNQwen3_5ForCausalLMConfig class for details.

    Examples:
        ```python
        from optimum.rbln.transformers.models.qwen3_5 import RBLNQwen3_5ForCausalLM

        model = RBLNQwen3_5ForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-0.8B",
            export=True,
            rbln_config={
                "num_devices": 1,
                "kvcache_partition_len": 4096,
                "max_seq_len": 8192,
                "device": 0,
            },
        )

        model.save_pretrained("compiled-qwen3.5-0.8b")
        ```
    """

    auto_model_class = AutoModelForCausalLM


class RBLNQwen3_5VisionModel(RBLNModel):
    """Qwen3.5 vision encoder for RBLN — a Qwen3-VL-style vision tower WITHOUT deepstack.

    The per-image window padding / rotary / position embedding interpolation helpers are defined here.
    """

    auto_model_class = None
    _supports_non_fp32 = True
    _tp_support = True

    def __post_init__(self, **kwargs):
        self.transformer = self.model[0]
        self.max_seq_len = torch.tensor(sorted(self.rbln_config.max_seq_len, reverse=False))
        config = self.config
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size * config.spatial_merge_size

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)
        # Precompute the rotary cos/sin tables up to the largest ViT bucket
        _freq_table = self.rotary_pos_emb(int(self.max_seq_len.max().item()))
        self.rotary_cos_table = _freq_table.cos()
        self.rotary_sin_table = _freq_table.sin()

        with no_init_weights():
            self.patch_embed = Qwen3_5VisionPatchEmbed(config=config)
            self.pos_embed = torch.nn.Embedding(config.num_position_embeddings, config.hidden_size)

        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.patch_embed.load_state_dict(artifacts["patch_embed"])
        self.pos_embed.load_state_dict(artifacts["pos_embed"])

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "Qwen3_5VisionModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNQwen3_5VisionModelConfig,
    ):
        save_dict = {}
        save_dict["patch_embed"] = model.patch_embed.state_dict()
        save_dict["pos_embed"] = model.pos_embed.state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNQwen3_5VisionModelConfig):
        return Qwen3_5VisionModelWrapper(model, rbln_config).eval()

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Qwen3_5VisionModel, __name)
        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors=None,
        model: Optional["PreTrainedModel"] = None,
        model_config: "PretrainedConfig" = None,
        rbln_config: RBLNQwen3_5VisionModelConfig | None = None,
    ) -> RBLNQwen3_5VisionModelConfig:
        hidden_size = model_config.hidden_size
        num_heads = model_config.num_heads
        head_dim = hidden_size // num_heads
        batch_size = rbln_config.batch_size

        input_infos = []
        for max_seq_len in rbln_config.max_seq_len:
            input_info = [
                ("hidden_states", [max_seq_len, hidden_size], rbln_config.dtype),
                ("attn_mask", [batch_size, 1, max_seq_len, max_seq_len], rbln_config.dtype),
                # cos/sin enter the device at fp32 and are cast to the device dtype inside the vision model
                ("cos", [batch_size, 1, max_seq_len, head_dim], torch.float32),
                ("sin", [batch_size, 1, max_seq_len, head_dim], torch.float32),
            ]
            input_infos.append(input_info)

        rbln_compile_config = RBLNCompileConfig(input_info=input_infos)
        rbln_config.set_compile_cfgs([rbln_compile_config])

        return rbln_config

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        merge_size = self.spatial_merge_size
        device = self.rotary_cos_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        # Gather cos/sin from the tables precomputed at object creation
        cos = self.rotary_cos_table[pos_ids].flatten(1)
        sin = self.rotary_sin_table[pos_ids].flatten(1)
        return cos, sin

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws, strict=False):  # noqa: B007
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws, strict=False)])

        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=False):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    @staticmethod
    def _pad_hidden_states(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        seq_len = hidden_states.shape[0]
        valid_len = seq_len

        if seq_len < max_seq_len:
            padding_size = max_seq_len - seq_len
            hidden_padding = torch.zeros(padding_size, hidden_states.shape[-1], dtype=hidden_states.dtype)
            hidden_states = torch.cat([hidden_states, hidden_padding], dim=0)

            cos, sin = position_embeddings
            pos_padding = torch.zeros(padding_size, cos.shape[-1], dtype=cos.dtype)
            cos = torch.cat([cos, pos_padding], dim=0)
            sin = torch.cat([sin, pos_padding], dim=0)
            position_embeddings = (cos, sin)

        attn_mask = torch.ones(1, 1, max_seq_len, max_seq_len, dtype=hidden_states.dtype)
        if valid_len < max_seq_len:
            attn_mask[:, :, valid_len:, :] = 0
            attn_mask[:, :, :, valid_len:] = 0

        return hidden_states, position_embeddings, attn_mask, valid_len

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states).to(self.rbln_config.dtype)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        cos, sin = self.rot_pos_emb(grid_thw)
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len, -1)
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        # fp32->device-dtype cast happens on-device in the vision wrapper
        position_embeddings = (cos, sin)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        num_images = len(cu_seqlens) - 1
        output_hidden_states = []
        for i in range(num_images):
            image_s, image_e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            image_hidden = hidden_states[image_s:image_e]
            image_cos = position_embeddings[0][image_s:image_e]
            image_sin = position_embeddings[1][image_s:image_e]

            image_seq_len = image_e - image_s
            try:
                ws_index = torch.searchsorted(self.max_seq_len, image_seq_len).item()
                max_seq_len = self.max_seq_len[ws_index].item()
            except Exception as e:
                raise ValueError(
                    f"Required seq_len({image_seq_len}) is larger than available "
                    f"max_seq_len({self.max_seq_len.tolist()})."
                ) from e

            image_hidden, (image_cos, image_sin), attn_mask, valid_len = self._pad_hidden_states(
                image_hidden, (image_cos, image_sin), max_seq_len
            )

            output = self.transformer(
                image_hidden,
                attn_mask,
                image_cos[None, None, :, :],
                image_sin[None, None, :, :],
            )
            main_output = output[0] if isinstance(output, (list, tuple)) else output
            merged_valid_len = valid_len // self.spatial_merge_unit
            output_hidden_states.append(main_output[:merged_valid_len])

        return torch.cat(output_hidden_states)


class RBLNQwen3_5Model(RBLNDecoderOnlyModel):
    auto_model_class = AutoModelForImageTextToText
    _decoder_wrapper_cls = Qwen3_5_LanguageModelWrapper
    _use_rotary_emb = False
    _rbln_submodules = [{"name": "visual"}]
    _config_class = Qwen3_5Config
    _rotary_emb_class = Qwen3_5TextRotaryEmbedding
    _get_rope_index_func = Qwen3_5Model.get_rope_index
    get_vision_position_ids = Qwen3_5Model.get_vision_position_ids

    @classmethod
    def _load_submodules(cls, model_save_dir, rbln_config, model=None, **kwargs):
        if model is None and not getattr(rbln_config, "_load_visual_runtime", True):
            return []
        return super()._load_submodules(model_save_dir, rbln_config, model=model, **kwargs)

    def __post_init__(self, **kwargs):
        if hasattr(self.config, "embedding_dim"):
            self.embedding_dim = self.config.embedding_dim
        if not isinstance(self.config.text_config, PretrainedConfig):
            self.config = self._config_class(
                text_config=self.config.text_config, vision_config=self.config.vision_config
            )
        super().__post_init__(**kwargs)
        self.visual = self.rbln_submodules[0] if self.rbln_submodules else None
        self.rotary_emb = self._rotary_emb_class(self.config.text_config)
        if not self.can_generate():
            self.block_tables = torch.arange(self.rbln_config.kvcache_num_blocks, dtype=torch.int16)

    @property
    def logits_last_dim(self):
        if self.can_generate():
            return self.config.text_config.vocab_size
        else:
            return self.embedding_dim if hasattr(self, "embedding_dim") else self.config.text_config.hidden_size

    def _create_embedding_layer(self):
        with no_init_weights():
            embed_tokens = torch.nn.Embedding(
                self.config.text_config.vocab_size,
                self.config.text_config.hidden_size,
                getattr(self.config.text_config, "pad_token_id", None),
            )
        return embed_tokens

    def _get_position_embeddings(self, hidden_states, position_ids):
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos = cos.unsqueeze(1).to(self.rbln_config.dtype)
        sin = sin.unsqueeze(1).to(self.rbln_config.dtype)
        return torch.stack([cos, sin])

    def setup_runtime(self):
        _qwen3_5_setup_hybrid_runtime(self)

    @classmethod
    def _update_rbln_config(cls, preprocessors=None, model=None, model_config=None, rbln_config=None):
        rbln_config.linear_attention_layers = _qwen3_5_linear_layer_indices(model_config)
        rbln_config = super()._update_rbln_config(
            preprocessors=preprocessors, model=model, model_config=model_config, rbln_config=rbln_config
        )
        if rbln_config.gdn_chunk_size is None:
            rbln_config.gdn_chunk_size = rbln_config.prefill_chunk_size
        if rbln_config.gdn_chunk_size > 128:
            raise ValueError(
                f"gdn_chunk_size must be <= 128, got {rbln_config.gdn_chunk_size}. "
                "Larger GatedDeltaNet sub-chunk sizes are not supported yet — "
                "set gdn_chunk_size to a value <= 128 that divides prefill_chunk_size."
            )
        return rbln_config

    @classmethod
    def get_input_info(cls, batch_size, query_length, rbln_config, model_config: PretrainedConfig):
        input_info = RBLNQwen3_5TextModel.get_input_info(batch_size, query_length, rbln_config, model_config)
        text_config = model_config.get_text_config()
        head_dim = getattr(text_config, "head_dim", None) or (
            text_config.hidden_size // text_config.num_attention_heads
        )
        rotary_ndims = int(head_dim * getattr(text_config, "partial_rotary_factor", 1.0))
        input_info.insert(3, ("position_emb", [2, batch_size, 1, query_length, rotary_ndims], rbln_config.dtype))
        return input_info

    def _preprocess_prefill(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None:
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            mask = input_ids == self.config.image_token_id
            mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            mask = input_ids == self.config.video_token_id
            mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, video_embeds)

        max_inputs_len = input_ids.shape[1]
        text_config = self.config.text_config
        head_dim = getattr(text_config, "head_dim", None) or (
            text_config.hidden_size // text_config.num_attention_heads
        )
        rotary_ndims = int(head_dim * getattr(text_config, "partial_rotary_factor", 1.0))
        all_position_embeds = torch.zeros(2, batch_size, 1, max_inputs_len, rotary_ndims, dtype=self.rbln_config.dtype)
        all_rope_deltas = []

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        image_idx, video_row_idx = 0, 0

        for b_idx in range(batch_size):
            input_id = input_ids[b_idx : b_idx + 1][:, attention_mask[b_idx].bool()]
            vision_start_indices = torch.argwhere(input_id == vision_start_token_id).squeeze(1)
            vision_tokens = input_id[0][vision_start_indices + 1]
            image_nums = int((vision_tokens == image_token_id).sum().item())
            video_nums = int((vision_tokens == video_token_id).sum().item())

            video_grid_slice = None
            if video_grid_thw is not None:
                start_row = video_row_idx
                consumed_video_chunks = 0
                while video_row_idx < video_grid_thw.shape[0] and consumed_video_chunks < video_nums:
                    consumed_video_chunks += int(video_grid_thw[video_row_idx, 0].item())
                    video_row_idx += 1
                video_grid_slice = video_grid_thw[start_row:video_row_idx]

            if mm_token_type_ids is not None:
                batch_mm_token_type_ids = mm_token_type_ids[b_idx : b_idx + 1][:, attention_mask[b_idx].bool()]
            else:
                batch_mm_token_type_ids = torch.zeros_like(input_id, dtype=torch.int)
                batch_mm_token_type_ids[input_id == image_token_id] = 1
                batch_mm_token_type_ids[input_id == video_token_id] = 2

            position_ids, rope_deltas = self._get_rope_index_func(
                input_id,
                batch_mm_token_type_ids,
                image_grid_thw[image_idx : image_idx + image_nums] if image_grid_thw is not None else None,
                video_grid_slice,
            )
            image_idx += image_nums

            position_embed = self._get_position_embeddings(inputs_embeds, position_ids)
            mask_indices = torch.nonzero(attention_mask[b_idx], as_tuple=True)[0]
            all_position_embeds[:, b_idx : b_idx + 1].index_copy_(dim=-2, index=mask_indices, source=position_embed)
            all_rope_deltas.append(rope_deltas)

        rope_deltas = torch.stack(all_rope_deltas)
        return inputs_embeds, all_position_embeds, rope_deltas

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> RBLNDecoderOnlyOutput:
        output_hidden_states = _validate_output_hidden_states(output_hidden_states, self.rbln_config)
        inputs_embeds, position_embed, rope_deltas = self._preprocess_prefill(
            input_ids,
            attention_mask,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        self.rope_deltas = rope_deltas
        batch_size, seq_len = inputs_embeds.shape[:2]

        text_config = self.config.get_text_config()
        all_hidden_states = (
            tuple(
                torch.zeros(batch_size, seq_len, text_config.hidden_size, dtype=self.rbln_config.dtype)
                for _ in range(text_config.num_hidden_layers + 1)
            )
            if output_hidden_states
            else None
        )
        logits = []
        for b_idx in range(batch_size):
            query_length = attention_mask[b_idx].sum(dim=-1).int().item() if attention_mask is not None else seq_len
            cache_position = torch.arange(query_length, dtype=torch.int32).unsqueeze(0)
            output = self.prefill_decoder(
                inputs_embeds=inputs_embeds[b_idx : b_idx + 1],
                attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                cache_position=cache_position,
                batch_idx=b_idx,
                position_embed=position_embed[:, b_idx : b_idx + 1],
                block_tables=self.block_tables,
            )
            logits.append(output.logits)
            if output_hidden_states:
                for l_idx in range(text_config.num_hidden_layers + 1):
                    all_hidden_states[l_idx][b_idx].copy_(output.hidden_states[l_idx][0])
        logits = torch.cat(logits, dim=0)

        if not return_dict:
            return logits
        return RBLNDecoderOnlyOutput(logits=logits, hidden_states=all_hidden_states)


class RBLNQwen3_5ForConditionalGeneration(RBLNQwen3_5Model, RBLNDecoderOnlyModelForCausalLM):
    """
    RBLNQwen3_5ForConditionalGeneration is a multi-modal model that integrates vision and language processing capabilities,
    optimized for RBLN NPUs. It is designed for conditional generation tasks that involve both image and text inputs.
    It pairs a vision encoder with the hybrid Qwen3.5 text backbone — GatedDeltaNet `linear_attention` layers interleaved
    with gated `full_attention` layers.

    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    Important Note:
        This model includes a Large Language Model (LLM). For optimal performance, it is highly recommended to use
        tensor parallelism for the language model. This can be achieved by using the `rbln_config` parameter in the
        `from_pretrained` method. Refer to the `from_pretrained` documentation and the RBLNQwen3_5ForConditionalGenerationConfig class for details.

    Examples:
        ```python
        from optimum.rbln import RBLNQwen3_5ForConditionalGeneration

        model = RBLNQwen3_5ForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3.5-27B",
            export=True,
            rbln_config={
                "visual": {
                    "num_devices": 8,
                    "max_seq_len": 16384,
                    "device": [0, 1, 2, 3, 4, 5, 6, 7],
                },
                "num_devices": 8,
                "kvcache_partition_len": 16384,
                "max_seq_len": 262144,
                "device": [0, 1, 2, 3, 4, 5, 6, 7],
            },
        )

        model.save_pretrained("qwen3.5-27B")
        ```
    """

    _supports_non_fp32 = True

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.rope_deltas = torch.zeros(self.rbln_config.batch_size)

    def can_generate(self):
        return True

    @classmethod
    def _get_compile_context(cls, compile_config, example_inputs):
        return _qwen3_5_build_compile_context(compile_config, example_inputs)

    @classmethod
    def _reconstruct_model_if_needed(cls, model: "PreTrainedModel"):
        model.model.lm_head = model.lm_head
        return model

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        generate_idx: torch.Tensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        model_inputs = {}
        is_prefill_phase = generate_idx is None
        if is_prefill_phase:
            generate_idx = attention_mask.sum(dim=-1, keepdim=True).int()
            cache_position = None
            model_inputs.update({"input_ids": input_ids})
        else:
            if inputs_embeds is not None:
                raise NotImplementedError("Specifying inputs_embeds in decoder phase is not supported.")
            input_ids = input_ids[:, -1:]
            cache_position = generate_idx
            generate_idx = generate_idx + 1
            mm_token_type_ids = None
            model_inputs.update({"input_ids": input_ids})

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "generate_idx": generate_idx,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "mm_token_type_ids": mm_token_type_ids,
            }
        )
        return model_inputs

    def _preprocess_decoder(
        self,
        input_ids: torch.LongTensor = None,
        cache_position: torch.LongTensor = None,
    ):
        if self.rbln_config.batch_size != cache_position.shape[0]:
            raise RuntimeError(
                f"Cache position size mismatch: got {cache_position.shape[0]}, expected {self.rbln_config.batch_size}."
            )

        inputs_embeds = self.embed_tokens(input_ids)
        position_embeds = []
        for b_idx in range(self.rbln_config.batch_size):
            delta = cache_position[b_idx] + self.rope_deltas[b_idx]
            position_ids = torch.arange(1).view(1, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            position_embed = self._get_position_embeddings(torch.zeros(1, dtype=self.rbln_config.dtype), position_ids)
            position_embeds.append(position_embed)

        position_embeds = torch.cat(position_embeds, dim=1)
        return inputs_embeds, position_embeds

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        generate_idx: torch.Tensor | None = None,
        return_dict: bool | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> RBLNDecoderOnlyOutput:
        output_hidden_states = _validate_output_hidden_states(output_hidden_states, self.rbln_config)
        text_config = self.config.get_text_config()
        if cache_position is None:  # prefill
            inputs_embeds, position_embed, rope_deltas = self._preprocess_prefill(
                input_ids,
                attention_mask,
                pixel_values,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
            batch_size, seq_len = inputs_embeds.shape[:2]

            all_hidden_states = (
                tuple(
                    torch.zeros(batch_size, seq_len, text_config.hidden_size, dtype=self.rbln_config.dtype)
                    for _ in range(text_config.num_hidden_layers + 1)
                )
                if output_hidden_states
                else None
            )
            logits = []
            for b_idx in range(batch_size):
                cache_pos = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)
                output = self.prefill_decoder(
                    inputs_embeds=inputs_embeds[b_idx : b_idx + 1],
                    attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                    cache_position=cache_pos,
                    batch_idx=b_idx,
                    position_embed=position_embed[:, b_idx : b_idx + 1],
                )
                logits.append(output.logits)
                if output_hidden_states:
                    for l_idx in range(text_config.num_hidden_layers + 1):
                        all_hidden_states[l_idx][b_idx].copy_(output.hidden_states[l_idx][0])
            logits = torch.cat(logits, dim=0)
        else:  # decode
            inputs_embeds, position_embed = self._preprocess_decoder(input_ids, cache_position)
            output = self.decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_embed=position_embed,
            )
            logits = output.logits
            all_hidden_states = output.hidden_states

        if not return_dict:
            return logits, generate_idx
        return RBLNDecoderOnlyOutput(logits=logits, generate_idx=generate_idx, hidden_states=all_hidden_states)
