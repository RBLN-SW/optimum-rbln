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

from typing import List, Optional

import torch
import torch.nn.functional as F

from ...modeling_outputs import RBLNDecoderOnlyOutput
from ..decoderonly.decoderonly_runtime_utils import RBLNRuntimeModel


class RBLNQwen3VLRuntimeModel(RBLNRuntimeModel):
    def __init__(self, *args, num_deepstack_layers: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_deepstack_layers = num_deepstack_layers

    def _prepare_prefill_inputs(
        self,
        inputs: torch.Tensor,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        visual_pos_mask: Optional[torch.Tensor] = None,
        deepstack_embeds: Optional[List[torch.Tensor]] = None,
    ):
        (
            inputs,
            cache_position,
            chunked_attention_mask,
            position_ids,
            position_embed,
            padded_cache_lengths,
            query_length,
            token_type_ids,
        ) = super()._prepare_prefill_inputs(
            inputs, cache_position, attention_mask, position_ids, position_embed, token_type_ids
        )

        padded_input_len = inputs.shape[1]
        if visual_pos_mask is not None:
            padding_size = padded_input_len - visual_pos_mask.shape[1]
            if padding_size > 0:
                visual_pos_mask = F.pad(visual_pos_mask, (0, padding_size), value=False)

        if deepstack_embeds is not None:
            padding_size = padded_input_len - deepstack_embeds.shape[1]
            if padding_size > 0:
                deepstack_embeds = F.pad(deepstack_embeds, (0, 0, 0, padding_size), value=0)

        return (
            inputs,
            cache_position,
            chunked_attention_mask,
            position_ids,
            position_embed,
            padded_cache_lengths,
            query_length,
            token_type_ids,
            visual_pos_mask,
            deepstack_embeds,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
        block_tables: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
        lora_int_ids: Optional[torch.Tensor] = None,
        visual_pos_mask: Optional[torch.Tensor] = None,
        deepstack_embeds: Optional[List[torch.Tensor]] = None,
    ):
        inputs = self.inputs_embeddings_if_needed(input_ids, inputs_embeds)
        block_tables, local_block_tables, is_external_block_tables = (
            self.page_table_manager.get_block_tables_if_needed(
                self.batch_size,
                cache_position,
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
            )
        else:
            return self.prefill_forward(
                inputs,
                cache_position,
                attention_mask,
                batch_idx,
                block_tables,
                is_external_block_tables=is_external_block_tables,
                position_embed=position_embed,
                token_type_ids=token_type_ids,
                local_block_tables=local_block_tables,
                lora_int_ids=lora_int_ids,
                visual_pos_mask=visual_pos_mask,
                deepstack_embeds=deepstack_embeds,
            )

    def prefill_forward(
        self,
        inputs: torch.Tensor,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
        block_tables: Optional[torch.Tensor] = None,
        is_external_block_tables: Optional[bool] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
        lora_int_ids: Optional[torch.Tensor] = None,
        visual_pos_mask: Optional[torch.Tensor] = None,
        deepstack_embeds: Optional[List[torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        if self.rbln_config.use_lora and lora_int_ids is None:
            if self.lora_int_ids is None:
                raise ValueError(
                    "lora_int_id is required when using LoRA. "
                    "You should call set_lora_int_ids() before forward() or pass lora_int_id to forward()."
                )

            if batch_idx is not None:
                lora_int_ids = self.lora_int_ids[batch_idx : batch_idx + 1].clone()
            else:
                lora_int_ids = self.lora_int_ids.clone()

        (
            inputs,
            cache_position,
            chunked_attention_mask,
            position_ids,
            position_embed,
            padded_cache_lengths,
            query_length,
            token_type_ids,
            visual_pos_mask,
            deepstack_embeds,
        ) = self._prepare_prefill_inputs(
            inputs,
            cache_position,
            attention_mask,
            position_ids,
            position_embed,
            token_type_ids=token_type_ids,
            visual_pos_mask=visual_pos_mask,
            deepstack_embeds=deepstack_embeds,
        )

        out_buffers, output_logits, output_hidden_states = self._prepare_prefill_outputs(query_length, attention_mask)
        prefix_cached_len = cache_position[0][0].item()

        padded_input_length = inputs.shape[1]
        for i, step in enumerate(range(0, padded_input_length, self.rbln_config.prefill_chunk_size)):
            chunk_size = self.rbln_config.prefill_chunk_size
            input_chunk = inputs[:, step : step + chunk_size]
            cache_pos_chunk = cache_position[:, step : step + chunk_size] + padded_cache_lengths
            position_embed_chunk = (
                position_embed[:, :, :, step : step + chunk_size, :] if position_embed is not None else None
            )
            position_ids_chunk = position_ids[:, step : step + chunk_size] if position_ids is not None else None

            chunk_visual_pos_mask = None
            chunk_deepstack_visual_embeds = None
            if visual_pos_mask is not None and deepstack_embeds is not None:
                chunk_visual_pos_mask = visual_pos_mask[:, step : step + chunk_size]
                chunk_deepstack_visual_embeds = deepstack_embeds[:, step : step + chunk_size, :]

            if chunked_attention_mask is not None:
                if self.rbln_config.use_position_ids:
                    valid_len = min(chunk_size, query_length - step)
                    for i in range(valid_len):
                        cache_idx = cache_pos_chunk[0, i].item()
                        if cache_idx < chunked_attention_mask.shape[-1]:
                            chunked_attention_mask[0, cache_idx] = 1
                else:
                    valid_len = min(chunk_size, query_length - step)
                    chunked_attention_mask[
                        :, :, :, prefix_cached_len + step : prefix_cached_len + step + valid_len
                    ] = 1

            if step + chunk_size >= query_length:
                # Last chunk
                query_position = torch.tensor(query_length - step - 1, dtype=torch.int16)
            else:
                query_position = torch.tensor(chunk_size - 1, dtype=torch.int16)

            forward_args = [
                input_chunk,
                cache_pos_chunk,
            ]

            if self.rbln_config.use_global_attention:
                forward_args.append(block_tables)

            forward_args.append(position_embed_chunk)

            if self.rbln_config.use_local_attention:
                forward_args.append(local_block_tables)

            if self.rbln_config.logits_to_keep > 0:
                forward_args.append(query_position)

            if self.rbln_config.use_attention_mask:
                forward_args.append(chunked_attention_mask)

            if chunk_visual_pos_mask is not None:
                forward_args.append(chunk_visual_pos_mask.to(torch.bool))
            else:
                forward_args.append(torch.zeros(1, chunk_size, dtype=torch.bool))

            if chunk_deepstack_visual_embeds is not None:
                forward_args.append(chunk_deepstack_visual_embeds)
            else:
                hidden_size = inputs.shape[-1]
                forward_args.append(
                    torch.zeros(self.num_deepstack_layers, chunk_size, hidden_size, dtype=inputs.dtype)
                )

            if self.rbln_config.use_position_ids:
                forward_args.append(position_ids_chunk)

            if self.rbln_config.use_lora:
                forward_args.append(lora_int_ids)

            super(RBLNRuntimeModel, self).forward(
                *forward_args,
                out=out_buffers[i],
            )

        if self.rbln_config.logits_to_keep > 0:
            output_logits = output_logits[:, :query_length, :]
        else:
            output_logits = output_logits[:, :query_length, :]
            if attention_mask is not None:
                new_output_logits = torch.full(
                    (1, attention_mask.shape[-1], output_logits.shape[-1]),
                    fill_value=1e-10,
                    dtype=output_logits.dtype,
                )
                mask_indices = torch.nonzero(attention_mask, as_tuple=True)[0]
                new_output_logits.index_copy_(dim=-2, index=mask_indices, source=output_logits)
                output_logits = new_output_logits

        if self.rbln_config.can_generate and not is_external_block_tables and self.rbln_config.use_attention_mask:
            if self.rbln_config.use_position_ids:
                self.dec_attn_mask[batch_idx : batch_idx + 1] = chunked_attention_mask
            else:
                self.dec_attn_mask[batch_idx].fill_(0)
                self.dec_attn_mask[batch_idx, :, :, :query_length] = 1

        if self.rbln_config.output_hidden_states:
            output_hidden_states = tuple(hs[:, :query_length, :] for hs in output_hidden_states)
            return RBLNDecoderOnlyOutput(logits=output_logits, hidden_states=output_hidden_states)
        else:
            return RBLNDecoderOnlyOutput(logits=output_logits, hidden_states=None)
