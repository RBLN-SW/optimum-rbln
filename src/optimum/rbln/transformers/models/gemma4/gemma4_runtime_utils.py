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

from typing import Any, Dict, Optional

import rebel
import torch

from ...modeling_outputs import RBLNDecoderOnlyOutput, RBLNGemma4ForCausalLMOutput
from ..decoderonly.decoderonly_runtime_utils import (
    RBLNDecoderOnlyChunkedMultimodalPrefillMixin,
    RBLNPytorchRuntime,
)
from ..decoderonly.modeling_decoderonly import RBLNRuntimeModel


class RBLNGemma4RuntimeModel(RBLNDecoderOnlyChunkedMultimodalPrefillMixin, RBLNRuntimeModel):
    # Extends the shared chunked-multimodal prefill mixin with two Gemma4-specific responsibilities:
    # 1. Holds host-side references to embed_tokens_per_layer so the per-layer residual stream can
    #    be computed from raw input_ids before the compiled graph is invoked.
    # 2. Inserts the precomputed per_layer_inputs tensor as the second positional argument in every
    #    runtime call, matching the argument order of Gemma4ForCausalLMWrapper.prepare_forward_args.

    _prefill_output_cls = RBLNGemma4ForCausalLMOutput

    def __init__(
        self,
        *args: Any,
        image_prefills: Optional[Dict[int, rebel.Runtime]] = None,
        embed_tokens_per_layer: Optional[torch.nn.Module] = None,
        num_hidden_layers: Optional[int] = None,
        hidden_size_per_layer_input: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embed_tokens_per_layer = embed_tokens_per_layer
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        # One runtime per image-prefill chunk bucket, keyed by chunk size; selected in `prefill_forward`.
        self.image_prefills = {
            chunk_size: RBLNPytorchRuntime(runtime) for chunk_size, runtime in (image_prefills or {}).items()
        }
        self.prefill = RBLNPytorchRuntime(self.runtime) if self.phase == "prefill" else None

    def compute_per_layer_inputs(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.hidden_size_per_layer_input or self.embed_tokens_per_layer is None:
            return None
        clamped_ids = input_ids.clamp(min=0, max=self.embed_tokens_per_layer.num_embeddings - 1)
        per_layer_inputs = self.embed_tokens_per_layer(clamped_ids)
        per_layer_inputs = per_layer_inputs.reshape(
            *clamped_ids.shape, self.num_hidden_layers, self.hidden_size_per_layer_input
        )
        return per_layer_inputs

    def _invoke_prefill_chunk(
        self,
        runtime,
        input_chunk: torch.Tensor,
        per_layer_chunk: Optional[torch.Tensor],
        cache_pos_chunk: torch.Tensor,
        block_tables: torch.Tensor,
        local_block_tables: Optional[torch.Tensor],
        query_position: torch.Tensor,
        chunked_attention_mask: torch.Tensor,
        position_ids_chunk: Optional[torch.Tensor],
        lora_int_ids: Optional[torch.Tensor],
    ):
        # Mirrors Gemma4ForCausalLMWrapper.prepare_forward_args: per_layer_inputs is the second
        # positional argument and an explicit None occupies the position_embed slot.
        return runtime(
            input_chunk,
            per_layer_chunk if self.hidden_size_per_layer_input else None,
            cache_pos_chunk,
            block_tables,
            local_block_tables,
            None,
            query_position,
            chunked_attention_mask,
            position_ids_chunk,
            lora_int_ids if self.rbln_config.use_lora else None,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        cache_position: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
        block_tables: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
        lora_int_ids: Optional[torch.Tensor] = None,
    ):
        inputs = self.inputs_embeddings_if_needed(input_ids, inputs_embeds)

        if per_layer_inputs is None and input_ids is not None:
            per_layer_inputs = self.compute_per_layer_inputs(input_ids)

        alloc_cache_position = cache_position
        if (
            self.phase != "decode"
            and cache_position is not None
            and self.rbln_config.use_image_prefill
            and token_type_ids is not None
        ):
            plan_token_type_ids = token_type_ids
            if attention_mask is not None:
                mask_bool = attention_mask.to(dtype=torch.bool)
                if mask_bool.dim() == 2:
                    mask_bool = mask_bool[0]
                plan_token_type_ids = token_type_ids[:, mask_bool]
            query_length = cache_position.shape[-1]
            _, _, alloc_len = self._plan_prefill_chunks(plan_token_type_ids, query_length)
            if alloc_len > self.rbln_config.max_seq_len:
                raise ValueError(
                    f"Chunked prefill requires {alloc_len} KV-cache slots (input length "
                    f"{query_length} plus {alloc_len - query_length} slots of trailing chunk_size "
                    f"write-extent overhang and partition-alignment padding), which exceeds max_seq_len "
                    f"({self.rbln_config.max_seq_len}). Increase max_seq_len or reduce the input length."
                )
            if alloc_len > query_length:
                extra_pos = int(cache_position[0, -1].item()) + (alloc_len - query_length)
                extra = cache_position.new_tensor([[extra_pos]])
                alloc_cache_position = torch.cat([cache_position, extra], dim=-1)

        block_tables, local_block_tables, is_external_block_tables = (
            self.page_table_manager.get_block_tables_if_needed(
                self.batch_size,
                alloc_cache_position,
                batch_idx=batch_idx,
                phase=self.phase,
                block_tables=block_tables,
                local_block_tables=local_block_tables,
            )
        )

        if self.phase == "decode":
            return self.decode_forward(
                inputs,
                cache_position,
                block_tables,
                is_external_block_tables,
                attention_mask=attention_mask,
                position_embed=position_embed,
                position_ids=position_ids,
                local_block_tables=local_block_tables,
                lora_int_ids=lora_int_ids,
                per_layer_inputs=per_layer_inputs,
            )
        else:
            return self.prefill_forward(
                inputs,
                cache_position,
                attention_mask,
                batch_idx,
                block_tables,
                is_external_block_tables=is_external_block_tables,
                position_ids=position_ids,
                position_embed=position_embed,
                token_type_ids=token_type_ids,
                local_block_tables=local_block_tables,
                lora_int_ids=lora_int_ids,
                per_layer_inputs=per_layer_inputs,
            )

    def decode_forward(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor = None,
        block_tables: torch.Tensor = None,
        is_external_block_tables: bool = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
        lora_int_ids: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if self.rbln_config.use_lora and lora_int_ids is None:
            if self.lora_int_ids is None:
                raise ValueError(
                    "lora_int_id is required when using LoRA. "
                    "You should call set_lora_int_ids() before forward() or pass lora_int_id to forward()."
                )
            lora_int_ids = self.lora_int_ids

        batch_size = inputs.shape[0]
        if batch_size != self.batch_size:
            raise RuntimeError(
                f"Batch size mismatch: got {batch_size}, expected {self.batch_size} (compiled batch size)."
            )
        if batch_size != cache_position.shape[0]:
            raise RuntimeError(f"Cache position size mismatch: got {cache_position.shape[0]}, expected {batch_size}.")

        if self.rbln_config.use_local_attention:
            local_block_tables = (
                local_block_tables
                if local_block_tables is not None
                else torch.arange(0, batch_size, dtype=torch.int16).view(batch_size, -1)
            )

        if self.rbln_config.use_attention_mask and attention_mask is None:
            for b_idx in range(batch_size):
                decoding_step = cache_position[b_idx].item()
                if not (0 <= decoding_step < self.dec_attn_mask.shape[-1]):
                    raise ValueError(
                        f"Decoding step {decoding_step} out of bounds for "
                        f"attention mask with shape {self.dec_attn_mask.shape}."
                    )
                self.dec_attn_mask[b_idx, decoding_step] = 1
                if self.batch_size < block_tables.shape[0]:
                    block_tables = block_tables[: self.batch_size]
                if self.dec_attn_mask is not None and self.batch_size < self.dec_attn_mask.shape[0]:
                    self.dec_attn_mask = self.dec_attn_mask[: self.batch_size]
            attention_mask = self.dec_attn_mask

        outputs = super(RBLNRuntimeModel, self).forward(
            inputs,
            per_layer_inputs if self.hidden_size_per_layer_input else None,
            cache_position,
            block_tables,
            local_block_tables,
            position_embed,
            attention_mask if self.rbln_config.use_attention_mask else None,
            position_ids if self.rbln_config.use_position_ids else None,
            lora_int_ids if self.rbln_config.use_lora else None,
            out=self.out_buffers,
        )

        if self.rbln_config.output_hidden_states:
            return RBLNDecoderOnlyOutput(logits=outputs[0], hidden_states=tuple(outputs[1:]))
        else:
            return RBLNDecoderOnlyOutput(logits=outputs, hidden_states=None)
