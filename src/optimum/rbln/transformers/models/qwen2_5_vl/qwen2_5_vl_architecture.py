import math
from typing import Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ..decoderonly.decoderonly_architecture import DecoderOnlyWrapper, apply_rotary_pos_emb
from .configuration_qwen2_5_vl import RBLNQwen2_5_VisionTransformerPretrainedModelConfig


class Qwen2_5_VisionTransformerWrapper(nn.Module):
    def __init__(self, model: torch.nn.Module, rbln_config: RBLNQwen2_5_VisionTransformerPretrainedModelConfig):
        super().__init__()
        self._original_mod = model
        self.fullatt_block_indexes = model.fullatt_block_indexes
        self.merger = model.merger
        self.rbln_config = rbln_config
        window_seq_len = (model.window_size // model.patch_size) ** 2
        self.blocks = self.wrap_vision_blocks(model.blocks, window_seq_len, rbln_config)

    def wrap_vision_blocks(
        self,
        blocks: torch.nn.ModuleList,
        window_seq_len: int,
        rbln_config: RBLNQwen2_5_VisionTransformerPretrainedModelConfig,
    ):
        wrapped_blocks = []
        for i, block in enumerate(blocks):
            is_full_attn = True if i in self.fullatt_block_indexes else False
            wrapped_blocks.append(Qwen2_5_VLVisionBlock(block, is_full_attn, window_seq_len, rbln_config))
        return nn.ModuleList(wrapped_blocks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        full_attn_masks: torch.Tensor,
        window_attn_masks: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        full_attn_masks = (1.0 - full_attn_masks) * torch.finfo(hidden_states.dtype).min
        window_attn_masks = (1.0 - window_attn_masks) * torch.finfo(hidden_states.dtype).min

        for i, block in enumerate(self.blocks):
            attn_masks = full_attn_masks if i in self.fullatt_block_indexes else window_attn_masks
            hidden_states = block(hidden_states, attn_masks, [cos, sin])

        hidden_states = self.merger(hidden_states)

        return hidden_states


class Qwen2_5_VLVisionBlock(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        is_full_attn: bool,
        window_seq_len: int,
        rbln_config: RBLNQwen2_5_VisionTransformerPretrainedModelConfig,
    ):
        super().__init__()
        self._origin_model = model
        self.rbln_config = rbln_config
        self.norm1 = model.norm1
        self.norm2 = model.norm2

        if is_full_attn:
            self.attn = Qwen2_5_VLVisionFullAttention(model.attn, rbln_config)
        else:
            self.attn = Qwen2_5_VLVisionWindowAttention(model.attn, window_seq_len, rbln_config)
        self.mlp = model.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_masks: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attn_masks,
            position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VLVisionFullAttention(nn.Module):
    def __init__(self, model: nn.Module, rbln_config: RBLNQwen2_5_VisionTransformerPretrainedModelConfig) -> None:
        super().__init__()
        self._origin_model = model
        self.rbln_config = rbln_config
        self.num_heads = model.num_heads
        self.head_dim = getattr(model, "head_dim", model.proj.in_features // model.num_heads)
        self.qkv = model.qkv
        self.proj = model.proj
        self.scale = torch.tensor(1 / math.sqrt(self.head_dim), dtype=rbln_config.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_masks: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        hidden_states = hidden_states.unsqueeze(0)
        q, k, v = (
            self.qkv(hidden_states).reshape(1, seq_length, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        )

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn_weights = attn_weights + attn_masks
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=hidden_states.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(1, seq_length, -1)
        attn_output = self.proj(attn_output).squeeze(0)

        return attn_output


class Qwen2_5_VLVisionWindowAttention(nn.Module):
    def __init__(
        self, model: nn.Module, window_seq_len: int, rbln_config: RBLNQwen2_5_VisionTransformerPretrainedModelConfig
    ) -> None:
        super().__init__()
        self._origin_model = model
        self.rbln_config = rbln_config
        self.num_heads = model.num_heads
        self.head_dim = getattr(model, "head_dim", model.proj.in_features // model.num_heads)
        self.qkv = model.qkv
        self.proj = model.proj
        self.window_seq_len = window_seq_len
        self.scale = torch.tensor(1 / math.sqrt(self.head_dim), dtype=rbln_config.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_masks: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        num_windows = seq_length // self.window_seq_len

        window_hidden_states = []
        for i in range(0, seq_length, self.window_seq_len):
            window_hidden_states.append(hidden_states[i : i + self.window_seq_len])
        hidden_states = torch.stack(window_hidden_states)

        q, k, v = (
            self.qkv(hidden_states)
            .reshape(num_windows, self.window_seq_len, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        cos, sin = position_embeddings
        cos = cos.reshape(num_windows, 1, seq_length // num_windows, -1)
        sin = sin.reshape(num_windows, 1, seq_length // num_windows, -1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale

        attn_weights = attn_weights + attn_masks
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(1, seq_length, -1)
        attn_output = self.proj(attn_output).squeeze(0)

        return attn_output


class Qwen2_5_VL_LanguageModelWrapper(DecoderOnlyWrapper):
    def get_decoder_layers(self, model: PreTrainedModel):
        return model.model.language_model.layers if hasattr(model, "model") else model.language_model.layers

    def get_model_layer(self, model: PreTrainedModel):
        return model.model.language_model if hasattr(model, "model") else model.language_model

    def prepare_forward_args(self, *args):
        args = list(args)
        input_ids = None if self.rbln_config.use_inputs_embeds else args.pop(0)
        inputs_embeds = args.pop(0) if self.rbln_config.use_inputs_embeds else None
        cache_position = args.pop(0)
        global_block_tables = args.pop(0)
        local_block_tables = None
        position_embeds = args.pop(0)
        query_position = args.pop(0) if self.phase == "prefill" and self.rbln_config.logits_to_keep > 0 else None
        position_ids = None
        attention_mask = args.pop(0) if self.rbln_config.use_attention_mask else None
        lora_int_id = args.pop(0) if self.rbln_config.lora_config else None
        past_key_values = args

        if len(past_key_values) != 2 * self.num_hidden_layers:
            raise ValueError(
                f"Different past_key_values to model's config. {len(past_key_values)} != {2 * self.num_hidden_layers}"
            )

        # [key, value] * n_layer -> ( (key, value) ) * n_layer
        # cache shape : batch, n_heads, 1, max_seq_len, head_dim
        _past_key_values = []
        for i in range(self.config.num_hidden_layers):
            key_states = past_key_values[i * 2]
            value_states = past_key_values[i * 2 + 1]
            past_key_value = [key_states, value_states]
            _past_key_values.append(past_key_value)
        past_key_values = _past_key_values

        return (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            query_position,
            attention_mask,
            position_ids,
            lora_int_id,
            past_key_values,
            position_embeds,
        )
