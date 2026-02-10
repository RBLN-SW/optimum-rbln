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
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
    apply_rotary_pos_emb,
    slice_and_unsqueeze_cos_sin,
)
from .configuration_qwen3_vl import RBLNQwen3VLVisionModelConfig


if TYPE_CHECKING:
    from .configuration_qwen3_vl import RBLNQwen3VLForConditionalGenerationConfig


class Qwen3VLVisionModelWrapper(nn.Module):
    def __init__(self, model: torch.nn.Module, rbln_config: RBLNQwen3VLVisionModelConfig):
        super().__init__()
        self.merger = model.merger
        self.rbln_config = rbln_config
        self.blocks = self.wrap_vision_blocks(model.blocks, rbln_config)
        self.deepstack_visual_indexes = model.deepstack_visual_indexes
        self.deepstack_merger_list = model.deepstack_merger_list

    def wrap_vision_blocks(
        self,
        blocks: torch.nn.ModuleList,
        rbln_config: RBLNQwen3VLVisionModelConfig,
    ):
        wrapped_blocks = []
        for block in blocks:
            wrapped_blocks.append(Qwen3VLVisionBlock(block, rbln_config))
        return nn.ModuleList(wrapped_blocks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        attn_mask = (1.0 - attn_mask) * torch.finfo(hidden_states.dtype).min

        deepstack_features = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, attn_mask, [cos, sin])
            if layer_num in self.deepstack_visual_indexes:
                deepstack_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[deepstack_idx](hidden_states)
                deepstack_features.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)
        return hidden_states, *deepstack_features


class Qwen3VLVisionBlock(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        rbln_config: RBLNQwen3VLVisionModelConfig,
    ):
        super().__init__()
        self._origin_model = model
        self.rbln_config = rbln_config
        self.norm1 = model.norm1
        self.norm2 = model.norm2
        self.attn = Qwen3VLVisionAttention(model.attn, rbln_config)
        self.mlp = model.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attn_mask,
            position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, model: nn.Module, rbln_config: RBLNQwen3VLVisionModelConfig) -> None:
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
        attn_mask: torch.Tensor,
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
        attn_weights = attn_weights + attn_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=hidden_states.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(1, seq_length, -1)
        attn_output = self.proj(attn_output).squeeze(0)

        return attn_output


class Qwen3VLDecoderOnlyForCausalLM(DecoderOnlyForCausalLM):
    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        query_position: torch.Tensor = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        rotary_emb: nn.Module = None,
        global_block_tables: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
        lora_int_id: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        visual_pos_mask: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[torch.Tensor] = None,
    ):
        hidden_states, all_hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
            lora_int_id=lora_int_id,
            output_hidden_states=output_hidden_states,
            visual_pos_mask=visual_pos_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        if "prefill" in self.phase and query_position is not None:
            hidden_states = hidden_states[:, query_position.to(torch.int).unsqueeze(0)]

        logits = self.lm_head(hidden_states)

        if getattr(self.config, "final_logit_softcapping", None) is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        return logits, all_hidden_states


class Qwen3VLDecoderOnlyModel(DecoderOnlyModel):
    def __init__(self, model, layers, rbln_config, num_deepstack_layers: int = 3, **kwargs):
        super().__init__(model, layers, rbln_config, **kwargs)
        self.num_deepstack_layers = num_deepstack_layers

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_mask: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        mask_expanded = visual_pos_mask.unsqueeze(-1).to(hidden_states.dtype)
        hidden_states = hidden_states + visual_embeds.unsqueeze(0) * mask_expanded
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        query_position: torch.Tensor = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        rotary_emb: Optional[Union[nn.Module, torch.Tensor]] = None,
        global_block_tables: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
        lora_int_id: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        visual_pos_mask: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[torch.Tensor] = None,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_embedding()(input_ids)

        hidden_states = inputs_embeds * self.hidden_multiplier

        position_ids = position_ids if position_ids is not None else cache_position
        if rotary_emb is not None:
            if isinstance(rotary_emb, torch.Tensor):
                cos = rotary_emb[0]
                sin = rotary_emb[1]
            else:
                cos, sin = rotary_emb(hidden_states, self.max_seq_len)
                cos, sin = slice_and_unsqueeze_cos_sin(cos, sin, position_ids)
        else:
            cos, sin = None, None

        if self.attn_impl == "flash_attn":
            seq_positions = cache_position[:, 0]
            seq_positions = self.convert_sequence_positions_for_flash_attn(
                seq_positions=seq_positions, max_seq_len=self.max_seq_len
            )
        else:
            seq_positions = cache_position[:, :1]

        swa_attn_mask = None
        sliding_cache_pos = None
        if len(self.sliding_window_layers) > 0:
            cache_seq_len, cache_offset, swa_attn_mask = self.get_swa_custom_op_args(position_ids, query_position)
            sliding_cache_pos = (cache_seq_len, cache_offset)

        apply_deepstack = deepstack_visual_embeds is not None and visual_pos_mask is not None

        all_hidden_states = () if output_hidden_states else None
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            is_sliding = True if layer_idx in self.sliding_window_layers else False
            is_sliding_decode = is_sliding and self.phase == "decode"
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=swa_attn_mask if is_sliding_decode else attention_mask,
                seq_positions=sliding_cache_pos if is_sliding else seq_positions,
                past_key_values=past_key_values,
                cos=cos,
                sin=sin,
                block_tables=local_block_tables if is_sliding else global_block_tables,
                lora_int_id=lora_int_id,
            )

            if apply_deepstack and layer_idx < self.num_deepstack_layers:
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_mask, deepstack_visual_embeds[layer_idx]
                )

        hidden_states = self.get_last_layernorm()(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, all_hidden_states


class Qwen3VL_LanguageModelWrapper(DecoderOnlyWrapper):
    """
    Wrapper for Qwen3VL language model that adapts it for RBLN compilation.

    This wrapper handles the special configuration structure of Qwen3VL where
    `model.config` is `Qwen3VLConfig` (containing both vision and text configs),
    but the parent `DecoderOnlyWrapper` expects a text-only config with attributes
    like `num_hidden_layers`. The config swap during `__init__` ensures compatibility.
    """

    def __init__(
        self, model: PreTrainedModel, rbln_config: "RBLNQwen3VLForConditionalGenerationConfig", use_rotary_emb: bool
    ):
        vision_config = model.config.vision_config if hasattr(model.config, "vision_config") else None
        if vision_config is not None:
            self.num_deepstack_layers = len(vision_config.deepstack_visual_indexes)
        else:
            self.num_deepstack_layers = 3

        # FIXME: change configuration to text_config for parent class initialization.
        original_config = model.config
        model.config = model.config.text_config
        super().__init__(model, rbln_config, use_rotary_emb)
        model.config = original_config

    def get_decoder_layers(self, model: PreTrainedModel):
        if hasattr(model, "language_model"):
            return model.language_model.layers
        elif hasattr(model, "model"):
            return model.model.layers
        return model.layers

    def get_model_layer(self, model: PreTrainedModel):
        if hasattr(model, "language_model"):
            return model.language_model
        elif hasattr(model, "model"):
            return model.model
        return model

    def get_rbln_attn_class(self):
        return Qwen3VLAttention

    def get_rbln_model_class(self):
        return Qwen3VLDecoderOnlyModel

    def get_rbln_causal_lm_class(self):
        return Qwen3VLDecoderOnlyForCausalLM

    def convert_to_rbln_class(self, model: PreTrainedModel, max_seq_len: int, use_rotary_emb: bool):
        new_layers = []
        for layer_idx, layer in enumerate(self.get_decoder_layers(model)):
            is_sliding = layer_idx in self.rbln_config.sliding_window_layers
            new_self_attn = self.get_rbln_attn_class()(
                self.get_attn_layer(layer), self.rbln_config, is_sliding=is_sliding
            )
            new_layer = self.get_rbln_layer_class()(layer, new_self_attn, lora_config=self.rbln_config.lora_config)
            new_layers.append(new_layer)

        new_model = self.get_rbln_model_class()(
            self.get_model_layer(model),
            new_layers,
            self.rbln_config,
            num_deepstack_layers=self.num_deepstack_layers,
            use_learned_pos_emb=self.__class__._use_learned_pos_emb,
            use_rotary_emb=use_rotary_emb,
        )

        if self.is_causal_lm:
            new_model = self.get_rbln_causal_lm_class()(model, new_model)
            return new_model
        else:
            return new_model

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

        if "prefill" in self.phase:
            visual_pos_mask = args.pop(0)
            deepstack_visual_embeds = args.pop(0)
        else:
            visual_pos_mask = None
            deepstack_visual_embeds = None

        past_key_values = args

        if len(past_key_values) != 2 * self.num_hidden_layers:
            raise ValueError(
                f"Different past_key_values to model's config. {len(past_key_values)} != {2 * self.num_hidden_layers}"
            )

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
            visual_pos_mask,
            deepstack_visual_embeds,
        )

    def forward(self, *args):
        (
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
            rotary_emb,
            visual_pos_mask,
            deepstack_visual_embeds,
        ) = self.prepare_forward_args(*args)

        logits, all_hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
            lora_int_id=lora_int_id,
            output_hidden_states=self.rbln_config.output_hidden_states,
            visual_pos_mask=visual_pos_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        if self.rbln_config.output_hidden_states:
            return logits, all_hidden_states
        else:
            return logits


class Qwen3VLAttention(DecoderOnlyAttention):
    def __post_init__(self, self_attn):
        self.q_proj = self_attn.q_proj
        self.k_proj = self_attn.k_proj
        self.v_proj = self_attn.v_proj
        self.o_proj = self_attn.o_proj
        self.q_norm = self_attn.q_norm
        self.k_norm = self_attn.k_norm

    def get_attn_scale(self, self_attn):
        if hasattr(self_attn, "config") and hasattr(self_attn.config, "head_dim"):
            return self_attn.config.head_dim**-0.5
        return self_attn.head_dim**-0.5
