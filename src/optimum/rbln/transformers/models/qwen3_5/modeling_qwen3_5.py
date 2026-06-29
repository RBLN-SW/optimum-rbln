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

import inspect
from typing import Any, Callable, Optional

import torch
from transformers import AutoModelForImageTextToText, PretrainedConfig, PreTrainedModel
from transformers.initialization import no_init_weights
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Model as HFQwen3_5Model,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionModel,
    Qwen3_5VisionPatchEmbed,
    Qwen3_5VisionRotaryEmbedding,
)

from ....configuration_utils import RBLNCompileConfig
from ....utils import logging
from ...modeling_outputs import RBLNDecoderOnlyOutput, _validate_output_hidden_states
from ..decoderonly.configuration_decoderonly import KVCacheMeta
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyModel, RBLNDecoderOnlyModelForCausalLM
from ..qwen3_vl.modeling_qwen3_vl import RBLNQwen3VLModel, RBLNQwen3VLVisionModel
from .configuration_qwen3_5 import (
    RBLNQwen3_5ForConditionalGenerationConfig,  # noqa: F401
    RBLNQwen3_5VisionModelConfig,  # noqa: F401
    RBLNQwen3_5VLModelConfig,  # noqa: F401
)
from .qwen3_5_architecture import Qwen3_5Wrapper
from .qwen3_5_vl_architecture import Qwen3_5VisionModelWrapper, Qwen3_5VL_LanguageModelWrapper


logger = logging.get_logger(__name__)


class RBLNQwen3_5ForCausalLM(RBLNDecoderOnlyModelForCausalLM):
    """
    RBLN wrapper for the Qwen3.5 text backbone (`Qwen3_5ForCausalLM`).

    Qwen3.5 is a hybrid decoder: `full_attention` layers use the standard paged KV cache, while
    `linear_attention` (GatedDeltaNet) layers carry a `conv_state` + `recurrent_state` instead.
    The two state tensors reuse the layer's two `past_key_values` slots positionally; this class
    overrides `get_input_info` to emit the right per-layer tensor specs and `_update_rbln_config`
    to derive `linear_attention_layers` from the HF `layer_types`.
    """

    _decoder_wrapper_cls = Qwen3_5Wrapper

    def forward(self, *args, **kwargs):
        kwargs["return_dict"] = True
        return super().forward(*args, **kwargs)

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors=None,
        model=None,
        model_config: Optional[PretrainedConfig] = None,
        rbln_config=None,
    ):
        text_config = model_config.get_text_config()
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            raise ValueError("Qwen3.5 requires `layer_types` in the model config.")
        rbln_config.linear_attention_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
        return super()._update_rbln_config(
            preprocessors=preprocessors, model=model, model_config=model_config, rbln_config=rbln_config
        )

    @classmethod
    def get_input_info(cls, batch_size, query_length, rbln_config, model_config: PretrainedConfig):
        text_config = model_config.get_text_config()
        num_attention_heads = getattr(text_config, "n_head", None) or text_config.num_attention_heads
        num_key_value_heads = getattr(text_config, "num_key_value_heads", None) or num_attention_heads
        num_hidden_layers = getattr(text_config, "n_layer", None) or text_config.num_hidden_layers
        hidden_size = getattr(text_config, "n_embd", None) or text_config.hidden_size
        head_dim = getattr(text_config, "head_dim", None) or hidden_size // num_attention_heads
        is_prefill = query_length > 1

        # ----- standard prefix (mirrors RBLNDecoderOnlyModelForCausalLM.get_input_info) -----
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
            if rbln_config.use_position_ids:
                input_info.append(("attention_mask", [batch_size, rbln_config.max_seq_len], rbln_config.dtype))
            else:
                input_info.append(
                    ("attention_mask", [batch_size, 1, query_length, rbln_config.max_seq_len], rbln_config.dtype)
                )
        if rbln_config.use_position_ids:
            input_info.append(("position_ids", [batch_size, query_length], "int32"))
        if rbln_config.use_lora:
            input_info.append(("lora_int_ids", [batch_size], "int32"))

        # ----- per-layer state: full -> (key, value) paged KV;  linear -> (conv_state, recurrent_state) -----
        linear_layers = set(rbln_config.linear_attention_layers)
        conv_dim = 2 * (text_config.linear_num_key_heads * text_config.linear_key_head_dim) + (
            text_config.linear_num_value_heads * text_config.linear_value_head_dim
        )
        conv_state_len = text_config.linear_conv_kernel_dim - 1

        kvcache_dtype = rbln_config.dtype
        if rbln_config.quantization and rbln_config.quantization.kv_caches == "fp8":
            kvcache_dtype = "float8_e4m3fn"

        kvcache_metas = []
        for layer_idx in range(num_hidden_layers):
            if layer_idx in linear_layers:
                # conv_state innermost dim = conv_dim (a multiple of 64, an RBLN alignment requirement)
                input_info.append(
                    (f"conv_state_{layer_idx}", [batch_size, conv_state_len, conv_dim], rbln_config.dtype)
                )
                input_info.append(
                    (
                        f"recurrent_state_{layer_idx}",
                        [
                            batch_size,
                            text_config.linear_num_value_heads,
                            text_config.linear_key_head_dim,
                            text_config.linear_value_head_dim,
                        ],
                        rbln_config.dtype,
                    )
                )
            else:
                for slot in range(2):
                    name = f"past_key_values_{layer_idx * 2 + slot}"
                    meta = KVCacheMeta.make(
                        name,
                        layer_idx,
                        num_key_value_heads,
                        head_dim,
                        RBLNCompileConfig.normalize_dtype(kvcache_dtype),
                        rbln_config,
                    )
                    kvcache_metas.append(meta)
                    input_info.append((name, meta.compile_shape, meta.dtype))

        if len(rbln_config.kvcache_metas) == 0:
            rbln_config.kvcache_metas.extend(kvcache_metas)

        return input_info


class RBLNQwen3_5Model(RBLNDecoderOnlyModel):
    """The bare Qwen3.5 text model (no LM head)."""

    _decoder_wrapper_cls = Qwen3_5Wrapper
    _use_rotary_emb = True


# ---------------------------------------------------------------------------------------------
# Vision-language (Qwen3.5-VL) -- Qwen3-VL-style vision encoder (NO deepstack) + hybrid text.
# Mirrors the Qwen2.5-VL RBLN flow (deepstack-free): the vision encoder's merged embeddings are
# scattered into `inputs_embeds`, mRoPE `position_emb` is precomputed on the host, and the hybrid
# Qwen3.5 text backbone runs the GatedDeltaNet linear layers + gated full-attention layers.
# ---------------------------------------------------------------------------------------------


class RBLNQwen3_5VisionModel(RBLNQwen3VLVisionModel):
    """Qwen3.5 vision encoder for RBLN — the Qwen3-VL vision tower WITHOUT deepstack.

    Qwen3.5 deletes deepstack from the Qwen3-VL vision model, so the wrapper returns only the
    merged image embeddings and `forward` does not collect per-layer deepstack features.
    """

    def __post_init__(self, **kwargs):
        self.transformer = self.model[0]
        self.max_seq_len = torch.tensor(sorted(self.rbln_config.max_seq_len, reverse=False))
        config = self.config
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size * config.spatial_merge_size

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)

        with no_init_weights():
            self.patch_embed = Qwen3_5VisionPatchEmbed(config=config)
            self.pos_embed = torch.nn.Embedding(config.num_position_embeddings, config.hidden_size)

        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.patch_embed.load_state_dict(artifacts["patch_embed"])
        self.pos_embed.load_state_dict(artifacts["pos_embed"])

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

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states).to(self.rbln_config.dtype)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos().to(self.rbln_config.dtype), emb.sin().to(self.rbln_config.dtype))

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
            # The Qwen3.5 vision wrapper returns only the merged embeddings (no deepstack tuple).
            main_output = output[0] if isinstance(output, (list, tuple)) else output
            merged_valid_len = valid_len // self.spatial_merge_unit
            output_hidden_states.append(main_output[:merged_valid_len])

        return torch.cat(output_hidden_states)


class RBLNQwen3_5VLModel(RBLNQwen3VLModel):
    """Bare Qwen3.5-VL model (vision encoder + hybrid text, no LM head)."""

    auto_model_class = AutoModelForImageTextToText
    _decoder_wrapper_cls = Qwen3_5VL_LanguageModelWrapper
    _use_rotary_emb = False
    _rbln_submodules = [{"name": "visual"}]
    _config_class = Qwen3_5Config
    _rotary_emb_class = Qwen3_5TextRotaryEmbedding
    _get_rope_index_func = HFQwen3_5Model.get_rope_index

    def __post_init__(self, **kwargs):
        if hasattr(self.config, "embedding_dim"):
            self.embedding_dim = self.config.embedding_dim
        if not isinstance(self.config.text_config, PretrainedConfig):
            self.config = self._config_class(
                text_config=self.config.text_config, vision_config=self.config.vision_config
            )
        # Qwen3.5 has NO deepstack: skip RBLNQwen3VLModel.__post_init__ (it reads
        # vision_config.deepstack_visual_indexes) and run the decoder-only setup directly.
        RBLNDecoderOnlyModel.__post_init__(self, **kwargs)
        self.visual = self.rbln_submodules[0] if self.rbln_submodules else None
        self.rotary_emb = self._rotary_emb_class(self.config.text_config)
        if not self.can_generate():
            self.block_tables = torch.arange(self.rbln_config.kvcache_num_blocks, dtype=torch.int16)

    @classmethod
    def _update_rbln_config(cls, preprocessors=None, model=None, model_config=None, rbln_config=None):
        text_config = model_config.get_text_config()
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            raise ValueError("Qwen3.5 requires `layer_types` in the model config.")
        rbln_config.linear_attention_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
        return super()._update_rbln_config(
            preprocessors=preprocessors, model=model, model_config=model_config, rbln_config=rbln_config
        )

    @classmethod
    def get_input_info(cls, batch_size, query_length, rbln_config, model_config: PretrainedConfig):
        # Hybrid per-layer state specs (conv_state/recurrent_state for linear layers, paged KV for
        # full layers) are identical to the text CausalLM; reuse that builder, then insert the
        # precomputed mRoPE position embedding at index 3 (after block_tables). No deepstack inputs.
        input_info = RBLNQwen3_5ForCausalLM.get_input_info(batch_size, query_length, rbln_config, model_config)
        text_config = model_config.get_text_config()
        head_dim = getattr(text_config, "head_dim", None) or (
            text_config.hidden_size // text_config.num_attention_heads
        )
        input_info.insert(3, ("position_emb", [2, batch_size, 1, query_length, head_dim], rbln_config.dtype))
        return input_info

    def _preprocess_prefill(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        inputs_embeds = self.embed_tokens(input_ids).to(self.rbln_config.dtype)

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
        all_position_embeds = torch.zeros(2, batch_size, 1, max_inputs_len, head_dim, dtype=self.rbln_config.dtype)
        all_rope_deltas = []

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        image_idx, video_idx = 0, 0

        for b_idx in range(batch_size):
            input_id = input_ids[b_idx : b_idx + 1][:, attention_mask[b_idx].bool()]
            vision_start_indices = torch.argwhere(input_id == vision_start_token_id).squeeze(1)
            vision_tokens = input_id[0][vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()

            if mm_token_type_ids is not None:
                batch_mm_token_type_ids = mm_token_type_ids[b_idx : b_idx + 1][:, attention_mask[b_idx].bool()]
            else:
                batch_mm_token_type_ids = torch.zeros_like(input_id, dtype=torch.int)
                batch_mm_token_type_ids[input_id == image_token_id] = 1
                batch_mm_token_type_ids[input_id == video_token_id] = 2

            # Qwen3.5 get_rope_index has no `second_per_grid_ts` (it separates videos by timestamps).
            position_ids, rope_deltas = self._get_rope_index_func(
                input_id,
                batch_mm_token_type_ids,
                image_grid_thw[image_idx : image_idx + image_nums] if image_grid_thw is not None else None,
                video_grid_thw[video_idx : video_idx + video_nums] if video_grid_thw is not None else None,
            )
            image_idx += image_nums
            video_idx += video_nums

            position_embed = self._get_position_embeddings(inputs_embeds, position_ids)
            mask_indices = torch.nonzero(attention_mask[b_idx], as_tuple=True)[0]
            all_position_embeds[:, b_idx : b_idx + 1].index_copy_(dim=-2, index=mask_indices, source=position_embed)
            all_rope_deltas.append(rope_deltas)

        rope_deltas = torch.stack(all_rope_deltas)
        return inputs_embeds, all_position_embeds, rope_deltas


# MRO: RBLNQwen3_5ForConditionalGeneration -> RBLNQwen3_5VLModel -> RBLNQwen3VLModel
#      -> RBLNDecoderOnlyModelForCausalLM -> RBLNDecoderOnlyModel -> RBLNModel
class RBLNQwen3_5ForConditionalGeneration(RBLNQwen3_5VLModel, RBLNDecoderOnlyModelForCausalLM):
    """
    Vision-language Qwen3.5: a Qwen3-VL-style vision encoder (no deepstack) feeding the hybrid
    Qwen3.5 text backbone (GatedDeltaNet `linear_attention` layers + gated `full_attention`).

    Note: the hybrid text backbone currently requires a fused GatedDeltaNet device op to compile
    end-to-end (the recurrent decompose does not compose with device full-attention in one graph;
    see the project memo). This class wires the full VL structure so it works once that op lands.
    """

    auto_model_class = AutoModelForImageTextToText
    _decoder_wrapper_cls = Qwen3_5VL_LanguageModelWrapper
    _supports_non_fp32 = True
    _use_rotary_emb = False
    _rbln_submodules = [{"name": "visual"}]

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.rope_deltas = torch.zeros(self.rbln_config.batch_size)

    def can_generate(self):
        return True

    @classmethod
    def _reconstruct_model_if_needed(cls, model: "PreTrainedModel"):
        model.model.lm_head = model.lm_head
        return model

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        generate_idx: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
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

        inputs_embeds = self.embed_tokens(input_ids).to(self.rbln_config.dtype)
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
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        generate_idx: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        **kwargs,
    ) -> RBLNDecoderOnlyOutput:
        output_hidden_states = _validate_output_hidden_states(output_hidden_states, self.rbln_config)

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
            batch_size = inputs_embeds.shape[0]

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
            logits = torch.cat(logits, dim=0)
        else:  # decode
            inputs_embeds, position_embed = self._preprocess_decoder(input_ids, cache_position)
            output = self.decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_embed=position_embed,
            )
            logits = output.logits

        if not return_dict:
            return logits, generate_idx
        return RBLNDecoderOnlyOutput(logits=logits, generate_idx=generate_idx)
