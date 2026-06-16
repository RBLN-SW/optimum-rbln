# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForImageTextToText, PretrainedConfig, PreTrainedModel
from transformers.initialization import no_init_weights
from transformers.models.exaone4_5.configuration_exaone4_5 import Exaone4_5_Config
from transformers.models.exaone4_5.modeling_exaone4_5 import (
    Exaone4_5_PatchEmbed,
    Exaone4_5_VisionModel,
    Exaone4_5_VisionRotaryEmbedding,
)

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...modeling_outputs import RBLNDecoderOnlyOutput, _validate_output_hidden_states
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyModel, RBLNDecoderOnlyModelForCausalLM
from .configuration_exaone4_5 import (
    RBLNExaone4_5_ForConditionalGenerationConfig,
    RBLNExaone4_5_VisionModelConfig,
)
from .exaone4_5_architecture import Exaone4_5LanguageModelWrapper, Exaone4_5VisionTransformerWrapper


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


# TODO: if needed, add back the MTP with compile option
def _disable_mtp(model_config):
    if model_config is None:
        return
    text_config = getattr(model_config, "text_config", model_config)
    if hasattr(text_config, "num_nextn_predict_layers"):
        text_config.num_nextn_predict_layers = 0
    if hasattr(text_config, "_num_mtp_layers"):
        text_config._num_mtp_layers = 0
    if hasattr(text_config, "layer_types") and hasattr(text_config, "num_hidden_layers"):
        text_config.layer_types = text_config.layer_types[: text_config.num_hidden_layers]


class RBLNExaone4_5_VisionModel(RBLNModel):
    """
    RBLN optimized EXAONE-4.5 vision transformer model.

    This class provides hardware-accelerated inference for the EXAONE-4.5 vision transformer
    on RBLN devices, supporting image and video encoding for multimodal vision-language tasks
    with window-based attention mechanisms.
    """

    auto_model_class = None
    _supports_non_fp32 = True
    _tp_support = True

    def __post_init__(self, **kwargs):
        self.transformer = self.model[0]
        self.max_seq_len = torch.tensor(sorted(self.rbln_config.max_seq_len, reverse=False))
        config = self.config.vision_config if hasattr(self.config, "vision_config") else self.config
        self.config = config
        self.window_size = config.window_size
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size * config.spatial_merge_size
        self.rotary_pos_emb = Exaone4_5_VisionRotaryEmbedding((config.hidden_size // config.num_heads) // 2)
        with no_init_weights():
            self.patch_embed = Exaone4_5_PatchEmbed(
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
        model: "Exaone4_5_VisionModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNExaone4_5_VisionModelConfig,
    ):
        save_dict = {"patch_embed": model.patch_embed.state_dict()}
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNExaone4_5_VisionModelConfig):
        return Exaone4_5VisionTransformerWrapper(model, rbln_config).eval()

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Exaone4_5_VisionModel, __name)
        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "PretrainedConfig" = None,
        rbln_config: Optional[RBLNExaone4_5_VisionModelConfig] = None,
    ) -> RBLNExaone4_5_VisionModelConfig:
        model_config = model_config.vision_config if hasattr(model_config, "vision_config") else model_config
        window_size = model_config.window_size
        patch_size = model_config.patch_size
        hidden_size = model_config.hidden_size
        num_heads = model_config.num_heads
        head_dim = hidden_size // num_heads
        window_seq_len = (window_size // patch_size) ** 2

        input_infos = []
        for max_seq_len in rbln_config.max_seq_len:
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
                ("cos", [1, 1, max_seq_len, head_dim], rbln_config.dtype),
                ("sin", [1, 1, max_seq_len, head_dim], rbln_config.dtype),
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
                padding = torch.zeros(padding_size, segment.shape[-1], dtype=segment.dtype)
                padding_pos = torch.zeros(padding_size, cos_segment.shape[-1], dtype=cos_segment.dtype)
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

        window_attn_masks = torch.zeros(
            max_seq_len // window_seq_len,
            1,
            window_seq_len,
            window_seq_len,
            dtype=hidden_states.dtype,
        )
        for i, valid_len in enumerate(window_valid_lengths):
            window_attn_masks[i, :, :valid_len, :valid_len] = 1

        return hidden_state_padded, cos_padded, sin_padded, window_attn_masks, window_valid_lengths

    @staticmethod
    def _pad_for_full_attn_layers(
        hidden_state_padded, cos_padded, sin_padded, max_seq_len, window_valid_lengths, window_seq_len
    ):
        if hidden_state_padded.shape[0] < max_seq_len:
            full_padding_size = max_seq_len - hidden_state_padded.shape[0]
            full_padding_hidden = torch.zeros(
                full_padding_size, hidden_state_padded.shape[-1], dtype=hidden_state_padded.dtype
            )
            hidden_state_full_padded = torch.cat([hidden_state_padded, full_padding_hidden], dim=0)
            full_padding_pos = torch.zeros(full_padding_size, cos_padded.shape[-1], dtype=cos_padded.dtype)
            cos_full_padded = torch.cat([cos_padded, full_padding_pos], dim=0)
            sin_full_padded = torch.cat([sin_padded, full_padding_pos], dim=0)
            window_valid_lengths.extend([0] * (max_seq_len // window_seq_len - len(window_valid_lengths)))
        else:
            hidden_state_full_padded = hidden_state_padded
            cos_full_padded = cos_padded
            sin_full_padded = sin_padded

        full_attn_masks = torch.ones(1, 1, max_seq_len, max_seq_len, dtype=hidden_state_padded.dtype)
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
        cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32)
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
            dim=0, dtype=torch.int32
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        num_images = len(cu_seqlens) - 1
        cu_window_seqlens = cu_window_seqlens.tolist()
        window_seq_len = (self.window_size // self.patch_size) ** 2
        output_hidden_states = []

        for i in range(num_images):
            image_s, image_e = cu_seqlens[i], cu_seqlens[i + 1]
            window_indice = cu_window_seqlens[cu_window_seqlens.index(image_s) : cu_window_seqlens.index(image_e) + 1]
            window_padded_len = (len(window_indice) - 1) * window_seq_len

            try:
                ws_index = torch.searchsorted(self.max_seq_len, window_padded_len).item()
                max_seq_len = self.max_seq_len[ws_index]
            except Exception as e:
                raise ValueError(
                    f"Required seq_len({window_padded_len}) is larger than available max_seq_len({self.max_seq_len.tolist()})."
                ) from e

            hidden_state_padded, cos_padded, sin_padded, window_attn_masks, window_valid_lengths = (
                self._pad_for_window_attn_layers(
                    window_indice, hidden_states, position_embeddings, window_seq_len, max_seq_len
                )
            )
            hidden_state_full_padded, cos_full_padded, sin_full_padded, full_attn_masks = (
                self._pad_for_full_attn_layers(
                    hidden_state_padded, cos_padded, sin_padded, max_seq_len, window_valid_lengths, window_seq_len
                )
            )

            output = self.transformer(
                hidden_state_full_padded,
                full_attn_masks,
                window_attn_masks,
                cos_full_padded[None, None, :, :],
                sin_full_padded[None, None, :, :],
            )

            depadded_output = []
            for j, valid_len in enumerate(window_valid_lengths):
                start = j * (window_seq_len // self.spatial_merge_unit)
                end = start + (valid_len // self.spatial_merge_unit)
                depadded_output.append(output[start:end])
            output = torch.cat(depadded_output, dim=0)
            output_hidden_states.append(output)

        hidden_states = torch.cat(output_hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states


class RBLNExaone4_5_Model(RBLNDecoderOnlyModel):
    auto_model_class = AutoModelForImageTextToText
    _decoder_wrapper_cls = Exaone4_5LanguageModelWrapper
    _supports_non_fp32 = True
    _rbln_submodule_prefix = "model"
    _rbln_submodules = [{"name": "visual"}]
    _config_class = Exaone4_5_Config

    def __post_init__(self, **kwargs):
        if not isinstance(self.config.text_config, PretrainedConfig):
            self.config = self._config_class(
                text_config=self.config.text_config, vision_config=self.config.vision_config
            )
        _disable_mtp(self.config)
        super().__post_init__(**kwargs)
        self.visual = self.rbln_submodules[0]
        if not self.can_generate():
            self.block_tables = torch.arange(self.rbln_config.kvcache_num_blocks, dtype=torch.int16)

    @property
    def logits_last_dim(self):
        text_config = self.config.text_config
        if self.can_generate():
            return text_config.vocab_size
        else:
            return text_config.hidden_size

    def _create_embedding_layer(self):
        with no_init_weights():
            embed_tokens = torch.nn.Embedding(
                self.config.text_config.vocab_size,
                self.config.text_config.hidden_size,
                self.config.text_config.pad_token_id,
            )
        return embed_tokens

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config):
        _disable_mtp(model.config)
        return super()._wrap_model_if_needed(model, rbln_config)

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional[PretrainedConfig] = None,
        rbln_config: Optional[RBLNExaone4_5_ForConditionalGenerationConfig] = None,
    ):
        if model is not None:
            _disable_mtp(model.config)
        text_config = model_config.text_config if hasattr(model_config, "text_config") else model_config
        _disable_mtp(text_config)
        return super()._update_rbln_config(
            preprocessors=preprocessors,
            model=model,
            model_config=text_config,
            rbln_config=rbln_config,
        )

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
        inputs_embeds = self.embed_tokens(input_ids).to(self.rbln_config.dtype)

        if pixel_values is not None:
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(mask, video_embeds)

        return inputs_embeds

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
        inputs_embeds = self._preprocess_prefill(
            input_ids,
            attention_mask,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
        )

        batch_size, seq_len = inputs_embeds.shape[:2]
        output_hidden_states = _validate_output_hidden_states(output_hidden_states, self.rbln_config)
        all_hidden_states = (
            tuple(
                torch.zeros(
                    batch_size,
                    seq_len,
                    self.config.text_config.hidden_size,
                    dtype=self.rbln_config.dtype,
                )
                for _ in range(self.config.text_config.num_hidden_layers + 1)
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
                block_tables=self.block_tables,
            )
            logits.append(output.logits)
            if self.rbln_config.output_hidden_states:
                for l_idx in range(self.config.text_config.num_hidden_layers + 1):
                    all_hidden_states[l_idx][b_idx].copy_(output.hidden_states[l_idx][0])
        logits = torch.cat(logits, dim=0)

        if not return_dict:
            return logits if not output_hidden_states else (logits, all_hidden_states)
        return (
            RBLNDecoderOnlyOutput(logits=logits, hidden_states=all_hidden_states)
            if output_hidden_states
            else RBLNDecoderOnlyOutput(logits=logits)
        )


class RBLNExaone4_5_ForConditionalGeneration(RBLNExaone4_5_Model, RBLNDecoderOnlyModelForCausalLM):
    """
    RBLNExaone4_5_ForConditionalGeneration is a multi-modal model that integrates vision and language
    processing capabilities, optimized for RBLN NPUs. It is designed for conditional generation tasks
    that involve both image and text inputs.

    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the
    generic methods the library implements for all its models.
    """

    auto_model_class = AutoModelForImageTextToText
    _decoder_wrapper_cls = Exaone4_5LanguageModelWrapper
    _supports_non_fp32 = True
    _rbln_submodules = [{"name": "visual"}]

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

        if cache_position is None:
            if generate_idx is None:
                if attention_mask is not None:
                    generate_idx = attention_mask.sum(dim=-1, keepdim=True).int()
                else:
                    generate_idx = torch.full((input_ids.shape[0], 1), input_ids.shape[1], dtype=torch.int32)

            inputs_embeds = self._preprocess_prefill(
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
                        self.config.text_config.hidden_size,
                        dtype=self.rbln_config.dtype,
                    )
                    for _ in range(self.config.text_config.num_hidden_layers + 1)
                )
                if output_hidden_states
                else None
            )
            logits = []
            for b_idx in range(batch_size):
                cache_position = torch.arange(0, generate_idx[b_idx].item(), dtype=torch.int32).unsqueeze(0)
                output = self.prefill_decoder(
                    inputs_embeds=inputs_embeds[b_idx : b_idx + 1],
                    attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                    cache_position=cache_position,
                    batch_idx=b_idx,
                )
                logits.append(output.logits)
                if self.rbln_config.output_hidden_states:
                    for l_idx in range(self.config.text_config.num_hidden_layers + 1):
                        all_hidden_states[l_idx][b_idx].copy_(output.hidden_states[l_idx][0])
            logits = torch.cat(logits, dim=0)
        else:
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("Either `input_ids` or `inputs_embeds` must be provided in decode phase.")
                inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds.to(self.rbln_config.dtype)
            output = self.decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
            )
            logits = output.logits
            all_hidden_states = output.hidden_states

        if not return_dict:
            return logits, generate_idx if not output_hidden_states else (logits, generate_idx, all_hidden_states)
        return RBLNDecoderOnlyOutput(
            logits=logits,
            generate_idx=generate_idx,
            hidden_states=all_hidden_states,
        )
