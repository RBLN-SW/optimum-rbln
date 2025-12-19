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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForVision2Seq,
    PretrainedConfig,
    PreTrainedModel,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.modeling_utils import no_init_weights
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLRotaryEmbedding,
)

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...modeling_outputs import RBLNDecoderOnlyOutput, _validate_output_hidden_states
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyModel, RBLNDecoderOnlyModelForCausalLM
from .configuration_qwen2_5_vl import (
    RBLNQwen2_5_VisionTransformerPretrainedModelConfig,
    RBLNQwen2_5_VLForConditionalGenerationConfig,
)
from .qwen2_5_vl_architecture import Qwen2_5_VisionTransformerWrapper, Qwen2_5_VL_LanguageModelWrapper


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


class RBLNQwen2_5_VisionTransformerPretrainedModel(RBLNModel):
    """
    RBLN optimized Qwen2.5-VL vision transformer model.

    This class provides hardware-accelerated inference for Qwen2.5-VL vision transformers
    on RBLN devices, supporting image and video encoding for multimodal vision-language tasks
    with window-based attention mechanisms.
    """

    auto_model_class = None
    _supports_non_fp32 = True

    def __post_init__(self, **kwargs):
        self.transformer = self.model[0]
        self.max_seq_lens = torch.tensor(sorted(self.rbln_config.max_seq_lens, reverse=False))
        config = self.config
        self.window_size = config.window_size
        self.patch_size = config.spatial_patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size * config.spatial_merge_size
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding((config.hidden_size // config.num_heads) // 2)
        with no_init_weights():
            self.patch_embed = Qwen2_5_VisionPatchEmbed(
                patch_size=config.patch_size,
                temporal_patch_size=config.temporal_patch_size,
                in_channels=config.in_channels,
                embed_dim=config.hidden_size,
            )
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.patch_embed.load_state_dict(artifacts["patch_embed"])

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "Qwen2_5_VLForConditionalGeneration",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNQwen2_5_VisionTransformerPretrainedModelConfig,
    ):
        save_dict = {}
        save_dict["patch_embed"] = model.patch_embed.state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _wrap_model_if_needed(
        cls, model: "PreTrainedModel", rbln_config: RBLNQwen2_5_VisionTransformerPretrainedModelConfig
    ):
        return Qwen2_5_VisionTransformerWrapper(model, rbln_config).eval()

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Qwen2_5_VisionTransformerPretrainedModel, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "PretrainedConfig" = None,
        rbln_config: Optional[RBLNQwen2_5_VisionTransformerPretrainedModelConfig] = None,
    ) -> RBLNQwen2_5_VisionTransformerPretrainedModelConfig:
        window_size = model_config.window_size
        patch_size = model_config.patch_size
        hidden_size = model_config.hidden_size
        num_heads = model_config.num_heads
        head_dim = hidden_size // num_heads
        window_seq_len = (window_size // patch_size) ** 2

        input_infos = []
        for max_seq_len in rbln_config.max_seq_lens:
            if max_seq_len % window_seq_len > 0:
                raise ValueError(
                    f"max_seq_len ({max_seq_len}) must be a multiple of window_seq_len ({window_seq_len})."
                )

            input_info = [
                ("hidden_states", [max_seq_len, hidden_size], rbln_config.dtype),
                ("full_attn_masks", [1, 1, max_seq_len, max_seq_len], rbln_config.dtype),
                (
                    "window_attn_masks",
                    [max_seq_len // window_seq_len, 1, window_seq_len, window_seq_len],
                    rbln_config.dtype,
                ),
                (
                    "cos",
                    [1, 1, max_seq_len, head_dim],
                    rbln_config.dtype,
                ),
                (
                    "sin",
                    [1, 1, max_seq_len, head_dim],
                    rbln_config.dtype,
                ),
            ]
            input_infos.append(input_info)

        rbln_compile_config = RBLNCompileConfig(input_info=input_infos)
        rbln_config.set_compile_cfgs([rbln_compile_config])

        return rbln_config

    @staticmethod
    def _pad_for_window_attn_layers(
        window_indice: List[int],
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        window_seq_len: int,
        max_seq_len: int,
    ):
        # Padding for Window Attention
        padded_hidden_state = []
        padded_cos = []
        padded_sin = []
        window_valid_lengths = []
        for i in range(len(window_indice) - 1):
            start, end = window_indice[i], window_indice[i + 1]
            segment = hidden_states[start:end]
            cos_segment = position_embeddings[0][start:end]
            sin_segment = position_embeddings[1][start:end]
            segment_len = end - start

            if segment_len < window_seq_len:
                padding_size = window_seq_len - segment_len
                padding = torch.zeros(
                    padding_size,
                    segment.shape[-1],
                    dtype=segment.dtype,
                )
                padding_pos = torch.zeros(
                    padding_size,
                    cos_segment.shape[-1],
                    dtype=cos_segment.dtype,
                )
                padded_segment = torch.cat([segment, padding], dim=0)
                padded_cos_segment = torch.cat([cos_segment, padding_pos], dim=0)
                padded_sin_segment = torch.cat([sin_segment, padding_pos], dim=0)
            else:
                padded_segment = segment
                padded_cos_segment = cos_segment
                padded_sin_segment = sin_segment
            padded_hidden_state.append(padded_segment)
            window_valid_lengths.append(segment_len)
            padded_cos.append(padded_cos_segment)
            padded_sin.append(padded_sin_segment)
        hidden_state_padded = torch.cat(padded_hidden_state)
        cos_padded = torch.cat(padded_cos, dim=0)
        sin_padded = torch.cat(padded_sin, dim=0)

        window_attn_masks = torch.ones(
            max_seq_len // window_seq_len,
            1,
            window_seq_len,
            window_seq_len,
            dtype=hidden_states.dtype,
        )
        for i, valid_len in enumerate(window_valid_lengths):
            if valid_len < window_seq_len:
                window_attn_masks[i, :, valid_len:, :] = 0
                window_attn_masks[i, :, :, valid_len:] = 0

        return hidden_state_padded, cos_padded, sin_padded, window_attn_masks, window_valid_lengths

    @staticmethod
    def _pad_for_full_attn_layers(
        hidden_state_padded, cos_padded, sin_padded, max_seq_len, window_valid_lengths, window_seq_len
    ):
        if hidden_state_padded.shape[0] < max_seq_len:
            full_padding_size = max_seq_len - hidden_state_padded.shape[0]
            full_padding_hidden = torch.zeros(
                full_padding_size,
                hidden_state_padded.shape[-1],
                dtype=hidden_state_padded.dtype,
            )
            hidden_state_full_padded = torch.cat([hidden_state_padded, full_padding_hidden], dim=0)  # [5120, 1280]
            full_padding_pos = torch.zeros(
                full_padding_size,
                cos_padded.shape[-1],
                dtype=cos_padded.dtype,
            )
            cos_full_padded = torch.cat([cos_padded, full_padding_pos], dim=0)
            sin_full_padded = torch.cat([sin_padded, full_padding_pos], dim=0)
            window_valid_lengths.extend([0] * (max_seq_len // window_seq_len - len(window_valid_lengths)))
        else:
            hidden_state_full_padded = hidden_state_padded
            cos_full_padded = cos_padded
            sin_full_padded = sin_padded

        full_attn_masks = torch.ones(
            1,
            1,
            max_seq_len,
            max_seq_len,
            dtype=hidden_state_padded.dtype,
        )
        for i, valid_len in enumerate(window_valid_lengths):
            start = i * window_seq_len
            end = start + window_seq_len
            full_attn_masks[:, :, start + valid_len : end, :] = 0
            full_attn_masks[:, :, :, start + valid_len : end] = 0

        return hidden_state_full_padded, cos_full_padded, sin_full_padded, full_attn_masks

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states).to(self.rbln_config.dtype)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos().to(self.rbln_config.dtype), emb.sin().to(self.rbln_config.dtype))

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        num_images = len(cu_seqlens) - 1
        cu_window_seqlens = cu_window_seqlens.tolist()
        window_seq_len = (self.window_size // self.patch_size) ** 2

        output_hidden_states = []

        # Process each image in the sequence
        for i in range(num_images):
            image_s, image_e = cu_seqlens[i], cu_seqlens[i + 1]
            window_indice = cu_window_seqlens[cu_window_seqlens.index(image_s) : cu_window_seqlens.index(image_e) + 1]

            # Select the nearest higher max_seq_len from the available compiled models.
            window_padded_len = len(window_indice) * window_seq_len
            try:
                ws_index = torch.searchsorted(self.max_seq_lens, window_padded_len).item()
                max_seq_len = self.max_seq_lens[ws_index]
            except Exception as e:
                raise ValueError(
                    f"Required seq_len({window_padded_len}) is larger than available max_seq_lens({self.max_seq_lens.tolist()})."
                ) from e

            # Padding for Window Attention Layers
            hidden_state_padded, cos_padded, sin_padded, window_attn_masks, window_valid_lengths = (
                self._pad_for_window_attn_layers(
                    window_indice, hidden_states, position_embeddings, window_seq_len, max_seq_len
                )
            )

            # Padding for Full Attention Layers
            hidden_state_full_padded, cos_full_padded, sin_full_padded, full_attn_masks = (
                self._pad_for_full_attn_layers(
                    hidden_state_padded, cos_padded, sin_padded, max_seq_len, window_valid_lengths, window_seq_len
                )
            )

            # RBLN run with the compiled model
            output = self.transformer(
                hidden_state_full_padded,
                full_attn_masks,
                window_attn_masks,
                cos_full_padded[None, None, :, :],
                sin_full_padded[None, None, :, :],
            )

            # Depadding
            depadded_output = []
            for i, valid_len in enumerate(window_valid_lengths):
                start = i * (window_seq_len // self.spatial_merge_unit)
                end = start + (valid_len // self.spatial_merge_unit)
                depadded_output.append(output[start:end])
            output = torch.cat(depadded_output, dim=0)

            output_hidden_states.append(output)
        hidden_states = torch.cat(output_hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


class RBLNQwen2_5_VLModel(RBLNDecoderOnlyModel):
    auto_model_class = AutoModelForVision2Seq
    _decoder_wrapper_cls = Qwen2_5_VL_LanguageModelWrapper
    _use_rotary_emb = False
    _rbln_submodules = [
        {"name": "visual"},
    ]
    _config_class = Qwen2_5_VLConfig
    _rotary_emb_class = Qwen2_5_VLRotaryEmbedding
    _get_rope_index_func = Qwen2_5_VLModel.get_rope_index

    def __post_init__(self, **kwargs):
        if hasattr(self.config, "embedding_dim"):
            self.embedding_dim = self.config.embedding_dim

        if not isinstance(self.config.text_config, PretrainedConfig):
            self.config = self._config_class(
                text_config=self.config.text_config, vision_config=self.config.vision_config
            )

        super().__post_init__(**kwargs)
        self.visual = self.rbln_submodules[0]
        self.rotary_emb = self._rotary_emb_class(self.config)
        if not self.can_generate():
            self.block_tables = torch.arange(self.rbln_config.kvcache_num_blocks, dtype=torch.int16)

    @property
    def logits_last_dim(self):
        if self.can_generate():
            return self.config.vocab_size
        else:
            return self.embedding_dim if hasattr(self, "embedding_dim") else self.config.hidden_size

    def _create_embedding_layer(self):
        with no_init_weights():
            embed_tokens = torch.nn.Embedding(
                self.config.text_config.vocab_size,
                self.config.text_config.hidden_size,
                self.config.text_config.pad_token_id,
            )
        return embed_tokens

    @classmethod
    def get_input_info(
        cls,
        batch_size: int,
        query_length: int,
        rbln_config: RBLNQwen2_5_VLForConditionalGenerationConfig,
        model_config: PretrainedConfig,
    ):
        input_info = super().get_input_info(batch_size, query_length, rbln_config, model_config)
        pos_idx = 3
        input_info.insert(
            pos_idx,
            (
                "position_emb",
                [2, batch_size, 1, query_length, model_config.hidden_size // model_config.num_attention_heads],
                rbln_config.dtype,
            ),
        )

        return input_info

    def _get_position_embeddings(self, hidden_states, position_ids):
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        mrope_section = self.config.rope_scaling["mrope_section"] * 2
        cos = (
            torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
            .unsqueeze(1)
            .to(self.rbln_config.dtype)
        )
        sin = (
            torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)
            .unsqueeze(1)
            .to(self.rbln_config.dtype)
        )
        return torch.stack([cos, sin])

    def _preprocess_prefill(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_grid_thw: torch.LongTensor = None,
        video_grid_thw: torch.LongTensor = None,
        second_per_grid_ts: torch.Tensor = None,
    ):
        batch_size = input_ids.shape[0]
        inputs_embeds = self.embed_tokens(input_ids).to(self.rbln_config.dtype)

        if pixel_values is not None:
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, video_embeds)

        max_inputs_len = input_ids.shape[1]

        head_dim = getattr(self.config, "head_dim", None) or self.config.hidden_size // self.config.num_attention_heads
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
            position_ids, rope_deltas = self._get_rope_index_func(
                input_id,
                image_grid_thw[image_idx : image_idx + image_nums] if image_grid_thw is not None else None,
                video_grid_thw[video_idx : video_idx + video_nums] if video_grid_thw is not None else None,
                second_per_grid_ts[video_idx : video_idx + video_nums] if second_per_grid_ts is not None else None,
            )
            image_idx += image_nums
            video_idx += video_nums

            position_embed = self._get_position_embeddings(inputs_embeds, position_ids)
            mask_indices = torch.nonzero(attention_mask[b_idx], as_tuple=True)[0]
            all_position_embeds[:, b_idx : b_idx + 1].index_copy_(dim=-2, index=mask_indices, source=position_embed)
            all_rope_deltas.append(rope_deltas)

        rope_deltas = torch.stack(all_rope_deltas)

        return inputs_embeds, all_position_embeds, rope_deltas

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
        second_per_grid_ts: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> RBLNDecoderOnlyOutput:
        inputs_embeds, position_embed, rope_deltas = self._preprocess_prefill(
            input_ids,
            attention_mask,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
        )

        self.rope_deltas = rope_deltas
        batch_size, seq_len = inputs_embeds.shape[:2]

        output_hidden_states = _validate_output_hidden_states(output_hidden_states, self.rbln_config)

        all_hidden_states = (
            tuple(
                torch.zeros(
                    batch_size,
                    seq_len,
                    self.config.hidden_size,
                    dtype=self.rbln_config.dtype,
                )
                for _ in range(self.config.num_hidden_layers + 1)
            )
            if output_hidden_states
            else None
        )

        logits = []
        for b_idx in range(batch_size):
            query_length = attention_mask[b_idx].sum(dim=-1).int().item()
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
            if self.rbln_config.output_hidden_states:
                for l_idx in range(self.config.num_hidden_layers + 1):
                    all_hidden_states[l_idx][b_idx].copy_(output.hidden_states[l_idx][0])
        logits = torch.cat(logits, dim=0)

        if not return_dict:
            return_value = logits if not output_hidden_states else (logits, all_hidden_states)
            return return_value
        else:
            return (
                RBLNDecoderOnlyOutput(logits=logits, hidden_states=all_hidden_states)
                if output_hidden_states
                else RBLNDecoderOnlyOutput(logits=logits)
            )


# MRO: RBLNQwen2_5_VLForConditionalGeneration -> RBLNQwen2_5_VLModel -> RBLNDecoderOnlyModelForCausalLM -> RBLNDecoderOnlyModel -> RBLNModel
class RBLNQwen2_5_VLForConditionalGeneration(RBLNQwen2_5_VLModel, RBLNDecoderOnlyModelForCausalLM):
    """
    RBLNQwen2_5_VLForConditionalGeneration is a multi-modal model that integrates vision and language processing capabilities,
    optimized for RBLN NPUs. It is designed for conditional generation tasks that involve both image and text inputs.

    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    Important Note:
        This model includes a Large Language Model (LLM). For optimal performance, it is highly recommended to use
        tensor parallelism for the language model. This can be achieved by using the `rbln_config` parameter in the
        `from_pretrained` method. Refer to the `from_pretrained` documentation and the RBLNQwen2_5_VLForConditionalGenerationConfig class for details.

    Examples:
        ```python
        from optimum.rbln import RBLNQwen2_5_VLForConditionalGeneration

        model = RBLNQwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            export=True,
            rbln_config={
                "visual": {
                    "max_seq_lens": 6400,
                    "device": 0,
                },
                "tensor_parallel_size": 8,
                "kvcache_partition_len": 16_384,
                "max_seq_len": 114_688,
                "device": [0, 1, 2, 3, 4, 5, 6, 7],
            },
        )

        model.save_pretrained("compiled-qwen2.5-vl-7b-instruct")
        ```
    """

    auto_model_class = AutoModelForVision2Seq
    _decoder_wrapper_cls = Qwen2_5_VL_LanguageModelWrapper
    _supports_non_fp32 = True
    _use_rotary_emb = False
    _rbln_submodules = [
        {"name": "visual"},
    ]

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
        second_per_grid_ts=None,
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
                "second_per_grid_ts": second_per_grid_ts,
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
        second_per_grid_ts: Optional[torch.Tensor] = None,
        generate_idx: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> RBLNDecoderOnlyOutput:
        output_hidden_states = _validate_output_hidden_states(output_hidden_states, self.rbln_config)
        # Prefill
        if cache_position is None:
            inputs_embeds, position_embed, rope_deltas = self._preprocess_prefill(
                input_ids,
                attention_mask,
                pixel_values,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
            )

            batch_size, seq_len = inputs_embeds.shape[:2]
            all_hidden_states = (
                tuple(
                    torch.zeros(
                        batch_size,
                        seq_len,
                        self.config.hidden_size,
                        dtype=self.rbln_config.dtype,
                    )
                    for _ in range(self.config.num_hidden_layers + 1)
                )
                if output_hidden_states
                else None
            )
            self.rope_deltas = rope_deltas

            logits = []
            for b_idx in range(batch_size):
                cache_position = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)

                output = self.prefill_decoder(
                    inputs_embeds=inputs_embeds[b_idx : b_idx + 1],
                    attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                    cache_position=cache_position,
                    batch_idx=b_idx,
                    position_embed=position_embed[:, b_idx : b_idx + 1],
                )
                logits.append(output.logits)
                if self.rbln_config.output_hidden_states:
                    for l_idx in range(self.config.num_hidden_layers + 1):
                        all_hidden_states[l_idx][b_idx].copy_(output.hidden_states[l_idx][0])
            logits = torch.cat(logits, dim=0)
        # Decoder
        else:
            inputs_embeds, position_embed = self._preprocess_decoder(input_ids, cache_position)
            output = self.decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_embed=position_embed,
            )
            logits = output.logits
            all_hidden_states = output.hidden_states

        if not return_dict:
            return_value = (
                logits,
                generate_idx if not output_hidden_states else (logits, generate_idx, all_hidden_states),
            )
            return return_value
        else:
            return RBLNDecoderOnlyOutput(
                logits=logits,
                generate_idx=generate_idx,
                hidden_states=all_hidden_states,
            )
