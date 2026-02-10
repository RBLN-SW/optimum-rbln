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
    Qwen3VLConfig,
)
from transformers.modeling_utils import no_init_weights
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionRotaryEmbedding,
)

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...modeling_outputs import RBLNDecoderOnlyOutput, _validate_output_hidden_states
from ..decoderonly.decoderonly_runtime_utils import RBLNPageTableManager, RBLNRuntimeModel
from ..decoderonly.modeling_decoderonly import RBLNDecoderOnlyModel, RBLNDecoderOnlyModelForCausalLM
from .configuration_qwen3_vl import (
    RBLNQwen3VLForConditionalGenerationConfig,
    RBLNQwen3VLVisionModelConfig,
)
from .qwen3_vl_architecture import Qwen3VL_LanguageModelWrapper, Qwen3VLVisionModelWrapper
from .qwen3_vl_runtime_utils import RBLNQwen3VLRuntimeModel


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


class RBLNQwen3VLVisionModel(RBLNModel):
    """
    RBLN optimized Qwen3-VL vision transformer model.

    This class provides hardware-accelerated inference for Qwen3-VL vision transformers
    on RBLN devices, supporting image and video encoding for multimodal vision-language tasks.
    """

    auto_model_class = None
    _supports_non_fp32 = True

    def __post_init__(self, **kwargs):
        self.transformer = self.model[0]
        self.max_seq_lens = torch.tensor(sorted(self.rbln_config.max_seq_lens, reverse=False))
        config = self.config
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size * config.spatial_merge_size

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes

        with no_init_weights():
            self.patch_embed = Qwen3VLVisionPatchEmbed(config=config)
            self.pos_embed = torch.nn.Embedding(config.num_position_embeddings, config.hidden_size)

        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.patch_embed.load_state_dict(artifacts["patch_embed"])
        self.pos_embed.load_state_dict(artifacts["pos_embed"])

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "Qwen3VLVisionModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNQwen3VLVisionModelConfig,
    ):
        save_dict = {}
        save_dict["patch_embed"] = model.patch_embed.state_dict()
        save_dict["pos_embed"] = model.pos_embed.state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNQwen3VLVisionModelConfig):
        return Qwen3VLVisionModelWrapper(model, rbln_config).eval()

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Qwen3VLVisionModel, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "PretrainedConfig" = None,
        rbln_config: Optional[RBLNQwen3VLVisionModelConfig] = None,
    ) -> RBLNQwen3VLVisionModelConfig:
        hidden_size = model_config.hidden_size
        num_heads = model_config.num_heads
        head_dim = hidden_size // num_heads

        input_infos = []
        for max_seq_len in rbln_config.max_seq_lens:
            input_info = [
                ("hidden_states", [max_seq_len, hidden_size], rbln_config.dtype),
                ("attn_mask", [1, 1, max_seq_len, max_seq_len], rbln_config.dtype),
                ("cos", [1, 1, max_seq_len, head_dim], rbln_config.dtype),
                ("sin", [1, 1, max_seq_len, head_dim], rbln_config.dtype),
            ]
            input_infos.append(input_info)

        rbln_compile_config = RBLNCompileConfig(input_info=input_infos)
        rbln_config.set_compile_cfgs([rbln_compile_config])

        return rbln_config

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

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

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

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
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
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

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
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
        all_deepstack_features = [[] for _ in range(len(self.deepstack_visual_indexes))]

        for i in range(num_images):
            image_s, image_e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            image_hidden = hidden_states[image_s:image_e]
            image_cos = position_embeddings[0][image_s:image_e]
            image_sin = position_embeddings[1][image_s:image_e]

            image_seq_len = image_e - image_s
            try:
                ws_index = torch.searchsorted(self.max_seq_lens, image_seq_len).item()
                max_seq_len = self.max_seq_lens[ws_index].item()
            except Exception as e:
                raise ValueError(
                    f"Required seq_len({image_seq_len}) is larger than available max_seq_lens({self.max_seq_lens.tolist()})."
                ) from e

            image_hidden, (image_cos, image_sin), attn_mask, valid_len = self._pad_hidden_states(
                image_hidden, (image_cos, image_sin), max_seq_len
            )

            outputs = self.transformer(
                image_hidden,
                attn_mask,
                image_cos[None, None, :, :],
                image_sin[None, None, :, :],
            )

            main_output = outputs[0]
            deepstack_outputs = outputs[1:]

            merged_valid_len = valid_len // self.spatial_merge_unit
            output_hidden_states.append(main_output[:merged_valid_len])

            for j, ds_output in enumerate(deepstack_outputs):
                all_deepstack_features[j].append(ds_output[:merged_valid_len])

        hidden_states = torch.cat(output_hidden_states)
        deepstack_features = [torch.cat(ds) for ds in all_deepstack_features]

        return hidden_states, deepstack_features


class RBLNQwen3VLModel(RBLNDecoderOnlyModel):
    auto_model_class = AutoModelForVision2Seq
    _decoder_wrapper_cls = Qwen3VL_LanguageModelWrapper
    _use_rotary_emb = False
    _rbln_submodules = [
        {"name": "visual"},
    ]
    _config_class = Qwen3VLConfig
    _rotary_emb_class = Qwen3VLTextRotaryEmbedding
    _get_rope_index_func = Qwen3VLModel.get_rope_index

    def __post_init__(self, **kwargs):
        if hasattr(self.config, "embedding_dim"):
            self.embedding_dim = self.config.embedding_dim

        if not isinstance(self.config.text_config, PretrainedConfig):
            self.config = self._config_class(
                text_config=self.config.text_config, vision_config=self.config.vision_config
            )

        self.num_deepstack_layers = len(self.config.vision_config.deepstack_visual_indexes)

        super().__post_init__(**kwargs)
        self.visual = self.rbln_submodules[0]
        self.rotary_emb = self._rotary_emb_class(self.config.text_config)
        if not self.can_generate():
            self.block_tables = torch.arange(self.rbln_config.kvcache_num_blocks, dtype=torch.int16)

    def setup_runtime(self):
        page_table_manager = RBLNPageTableManager(self.rbln_config)
        if self.rbln_config.use_position_ids:
            dec_attn_mask = torch.zeros(self.rbln_config.batch_size, self.rbln_config.max_seq_len, dtype=self.dtype)
        else:
            dec_attn_mask = torch.zeros(
                self.rbln_config.batch_size, 1, 1, self.rbln_config.max_seq_len, dtype=self.dtype
            )

        common_kwargs = {
            "main_input_name": "inputs_embeds" if self.rbln_config.use_inputs_embeds else "input_ids",
            "embed_tokens": self.embed_tokens,
            "dec_attn_mask": dec_attn_mask,
            "page_table_manager": page_table_manager,
            "rbln_config": self.rbln_config,
            "config": self.config.text_config,
        }

        self.prefill_decoder = RBLNQwen3VLRuntimeModel(
            runtime=self.model[0],
            phase="prefill",
            batch_size=self.rbln_config.batch_size,
            logits_last_dim=self.logits_last_dim,
            num_deepstack_layers=self.num_deepstack_layers,
            **common_kwargs,
        )

        if self.can_generate():
            self.decoders = {}
            for i, batch_size in enumerate(self.rbln_config.decoder_batch_sizes):
                self.decoders[batch_size] = RBLNRuntimeModel(
                    runtime=self.model[i + 1],
                    phase="decode",
                    batch_size=batch_size,
                    **common_kwargs,
                )

            self.decoder = self.decoders[self.rbln_config.batch_size]

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

    @classmethod
    def get_input_info(
        cls,
        batch_size: int,
        query_length: int,
        rbln_config: RBLNQwen3VLForConditionalGenerationConfig,
        model_config: PretrainedConfig,
    ):
        input_info = super().get_input_info(batch_size, query_length, rbln_config, model_config.text_config)

        pos_idx = 3
        head_dim = getattr(model_config.text_config, "head_dim", None) or (
            model_config.text_config.hidden_size // model_config.text_config.num_attention_heads
        )
        input_info.insert(
            pos_idx,
            (
                "position_emb",
                [2, batch_size, 1, query_length, head_dim],
                rbln_config.dtype,
            ),
        )

        is_prefill = query_length > 1
        if is_prefill:
            num_deepstack_layers = len(model_config.vision_config.deepstack_visual_indexes)
            hidden_size = model_config.text_config.hidden_size

            insert_idx = None

            for i, info in enumerate(input_info):
                if info[0] == "attention_mask":
                    insert_idx = i + 1
                    break

            if insert_idx is None:
                for i, info in enumerate(input_info):
                    if info[0] == "position_ids":
                        insert_idx = i
                        break

            if insert_idx is None:
                for i, info in enumerate(input_info):
                    if info[0].startswith("past_key_values"):
                        insert_idx = i
                        break

            if insert_idx is None:
                insert_idx = len(input_info)

            input_info.insert(insert_idx, ("visual_pos_mask", [batch_size, query_length], "bool"))
            input_info.insert(
                insert_idx + 1,
                ("deepstack_visual_embeds", [num_deepstack_layers, query_length, hidden_size], rbln_config.dtype),
            )

        return input_info

    def _get_position_embeddings(self, hidden_states, position_ids):
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos = cos.unsqueeze(1).to(self.rbln_config.dtype)
        sin = sin.unsqueeze(1).to(self.rbln_config.dtype)
        return torch.stack([cos, sin])

    def _preprocess_prefill(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_grid_thw: torch.LongTensor = None,
        video_grid_thw: torch.LongTensor = None,
    ):
        batch_size = input_ids.shape[0]
        inputs_embeds = self.embed_tokens(input_ids).to(self.rbln_config.dtype)

        deepstack_image_embeds = None
        deepstack_video_embeds = None
        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            image_mask = input_ids == self.config.image_token_id
            mask_unsqueezed = image_mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            video_mask = input_ids == self.config.video_token_id
            mask_unsqueezed = video_mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, video_embeds)

        visual_pos_mask, deepstack_visual_embeds = self._prepare_deepstack(
            image_mask, video_mask, deepstack_image_embeds, deepstack_video_embeds
        )

        max_inputs_len = input_ids.shape[1]

        head_dim = getattr(self.config.text_config, "head_dim", None) or (
            self.config.text_config.hidden_size // self.config.text_config.num_attention_heads
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
            position_ids, rope_deltas = self._get_rope_index_func(
                input_id,
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

        if deepstack_visual_embeds is not None:
            hidden_size = self.config.text_config.hidden_size
            num_layers = len(deepstack_visual_embeds)
            padded_deepstack = torch.zeros(
                num_layers, max_inputs_len, hidden_size, dtype=self.rbln_config.dtype, device=inputs_embeds.device
            )
            if visual_pos_mask is not None:
                visual_indices = torch.nonzero(visual_pos_mask[0], as_tuple=True)[0]
                for layer_idx, layer_embed in enumerate(deepstack_visual_embeds):
                    padded_deepstack[layer_idx, visual_indices] = layer_embed.to(padded_deepstack.dtype)
            deepstack_visual_embeds = padded_deepstack

        return inputs_embeds, all_position_embeds, rope_deltas, visual_pos_mask, deepstack_visual_embeds

    def _prepare_deepstack(
        self,
        image_mask: Optional[torch.Tensor],
        video_mask: Optional[torch.Tensor],
        deepstack_image_embeds: Optional[List[torch.Tensor]],
        deepstack_video_embeds: Optional[List[torch.Tensor]],
    ):
        visual_pos_mask = None
        deepstack_visual_embeds = None

        if image_mask is not None and video_mask is not None:
            visual_pos_mask = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_mask]
            video_mask_joint = video_mask[visual_pos_mask]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds, strict=False):
                embed_joint = img_embed.new_zeros(visual_pos_mask.sum(), img_embed.shape[-1])
                embed_joint[image_mask_joint] = img_embed
                embed_joint[video_mask_joint] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            visual_pos_mask = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_mask = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        return visual_pos_mask, deepstack_visual_embeds

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
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> RBLNDecoderOnlyOutput:
        inputs_embeds, position_embed, rope_deltas, visual_pos_mask, deepstack_visual_embeds = (
            self._preprocess_prefill(
                input_ids,
                attention_mask,
                pixel_values,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
            )
        )

        self.rope_deltas = rope_deltas
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
            query_length = attention_mask[b_idx].sum(dim=-1).int().item()
            cache_position = torch.arange(query_length, dtype=torch.int32).unsqueeze(0)

            output = self.prefill_decoder(
                inputs_embeds=inputs_embeds[b_idx : b_idx + 1],
                attention_mask=attention_mask[b_idx] if attention_mask is not None else None,
                cache_position=cache_position,
                batch_idx=b_idx,
                position_embed=position_embed[:, b_idx : b_idx + 1],
                block_tables=self.block_tables,
                visual_pos_mask=visual_pos_mask[b_idx : b_idx + 1] if visual_pos_mask is not None else None,
                deepstack_embeds=deepstack_visual_embeds,
            )
            logits.append(output.logits)
            if self.rbln_config.output_hidden_states:
                for l_idx in range(self.config.text_config.num_hidden_layers + 1):
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


class RBLNQwen3VLForConditionalGeneration(RBLNQwen3VLModel, RBLNDecoderOnlyModelForCausalLM):
    """
    RBLNQwen3VLForConditionalGeneration is a multi-modal model that integrates vision and language processing capabilities,
    optimized for RBLN NPUs. It is designed for conditional generation tasks that involve both image and text inputs.

    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    Important Note:
        This model includes a Large Language Model (LLM). For optimal performance, it is highly recommended to use
        tensor parallelism for the language model. This can be achieved by using the `rbln_config` parameter in the
        `from_pretrained` method. Refer to the `from_pretrained` documentation and the RBLNQwen3VLForConditionalGenerationConfig class for details.

    Examples:
        ```python
        from optimum.rbln import RBLNQwen3VLForConditionalGeneration

        model = RBLNQwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            export=True,
            rbln_config={
                "visual": {
                    "max_seq_lens": 6400,
                    "device": 0,
                },
                "tensor_parallel_size": 8,
                "kvcache_partition_len": 16_384,
                "max_seq_len": 262_144,
                "device": [0, 1, 2, 3, 4, 5, 6, 7],
            },
        )

        model.save_pretrained("compiled-qwen3-vl-4b-instruct")
        ```
    """

    auto_model_class = AutoModelForVision2Seq
    _decoder_wrapper_cls = Qwen3VL_LanguageModelWrapper
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
        **kwargs,
    ) -> RBLNDecoderOnlyOutput:
        output_hidden_states = _validate_output_hidden_states(output_hidden_states, self.rbln_config)
        # Prefill
        if cache_position is None:
            inputs_embeds, position_embed, rope_deltas, visual_pos_mask, deepstack_visual_embeds = (
                self._preprocess_prefill(
                    input_ids,
                    attention_mask,
                    pixel_values,
                    pixel_values_videos,
                    image_grid_thw,
                    video_grid_thw,
                )
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
                    visual_pos_mask=visual_pos_mask[b_idx : b_idx + 1] if visual_pos_mask is not None else None,
                    deepstack_embeds=deepstack_visual_embeds,
                )
                logits.append(output.logits)
                if self.rbln_config.output_hidden_states:
                    for l_idx in range(self.config.text_config.num_hidden_layers + 1):
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
