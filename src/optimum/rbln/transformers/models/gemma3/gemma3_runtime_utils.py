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
from typing import Optional

import rebel
import torch

from ...modeling_outputs import RBLNGemma3ForCausalLMOutput
from ..decoderonly.decoderonly_runtime_utils import RBLNPytorchRuntime
from ..decoderonly.modeling_decoderonly import RBLNRuntimeModel


class RBLNGemma3RuntimeModel(RBLNRuntimeModel):
    def __init__(self, *args, image_prefill: Optional[rebel.Runtime] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_prefill = RBLNPytorchRuntime(image_prefill)  # FIXME(taehoon)
        self.prefill = RBLNPytorchRuntime(self.runtime) if self.phase == "prefill" else None  # FIXME

    def _prepare_prefill_inputs(self, *args, **kwargs):
        (
            inputs,
            cache_position,
            chunked_attention_mask,
            position_ids,
            position_embed,
            padded_cache_lengths,
            query_length,
            token_type_ids,
        ) = super()._prepare_prefill_inputs(*args, **kwargs)

        # chunked_attention_mask shape
        chunked_attention_mask = torch.zeros(1, chunked_attention_mask.shape[-1], dtype=torch.float32)

        # In case of Gemma3ForConditionalGeneration, the loop counter may not be a prefill_chunk_size,
        # so we cannot guarantee that the last chunk starts at a position that is a multiple of prefill_chunk_size.
        if self.rbln_config.use_image_prefill:
            padding_size = self.rbln_config.image_prefill_chunk_size
            inputs = torch.nn.functional.pad(inputs, (0, 0, 0, padding_size))
            cache_position = torch.nn.functional.pad(cache_position, (0, padding_size))
            position_ids = torch.nn.functional.pad(position_ids, (0, padding_size))
            token_type_ids = torch.nn.functional.pad(token_type_ids, (0, padding_size), value=-1)

        return (
            inputs,
            cache_position,
            chunked_attention_mask,
            position_ids,
            position_embed,
            padded_cache_lengths,
            query_length,
            token_type_ids,
        )

    def prefill_forward(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        batch_idx: int = None,
        block_tables: torch.Tensor = None,
        is_external_block_tables: bool = None,
        position_embed: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
        lora_int_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Performs chunked prefill for efficient KV-cache updates and memory optimization.
        Instead of processing the entire sequence at once, the input is divided into chunks of size `prefill_chunk_size`,
        and each chunk is processed sequentially. This allows for better memory utilization and compatibility with continuous batching.
        """
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
        ) = self._prepare_prefill_inputs(
            inputs, cache_position, attention_mask, position_embed, token_type_ids=token_type_ids
        )

        step = 0
        output_logits = []
        all_hidden_states = [] if self.rbln_config.output_hidden_states else None
        while step < query_length:
            if self.rbln_config.use_image_prefill:
                # Check if the prefill chunk is an image prefill
                is_image_prefill = torch.all(
                    token_type_ids[:, step : step + self.rbln_config.image_prefill_chunk_size] == 1
                )
                # Check if the prefill chunk is a text prefill which have image_tokens in it.
                is_text_prefill_with_image_tokens = not is_image_prefill and torch.any(
                    token_type_ids[:, step : step + self.rbln_config.prefill_chunk_size] == 1
                )
            else:
                is_image_prefill, is_text_prefill_with_image_tokens = False, False

            # Check if the prefill chunk is the last chunk
            is_last_chunk = step + self.rbln_config.prefill_chunk_size >= query_length

            input_chunk = inputs[:, step : step + self.rbln_config.prefill_chunk_size]
            cache_pos_chunk = (
                cache_position[:, step : step + self.rbln_config.prefill_chunk_size] + padded_cache_lengths
            )
            position_ids_chunk = position_ids[:, step : step + self.rbln_config.prefill_chunk_size]

            # if text_prefill end with image_tokens, we only treat the text part.
            num_processed_tokens = self.rbln_config.prefill_chunk_size
            current_padded_cache_lengths = 0
            if is_text_prefill_with_image_tokens:
                first_image_token_idx = torch.where(
                    token_type_ids[:, step : step + self.rbln_config.prefill_chunk_size] == 1
                )[1][0]
                num_processed_tokens = first_image_token_idx.item()
                current_padded_cache_lengths = self.rbln_config.prefill_chunk_size - num_processed_tokens
            if is_last_chunk:
                num_processed_tokens = query_length - step

            chunked_attention_mask[
                :, step + padded_cache_lengths : step + num_processed_tokens + padded_cache_lengths
            ] = 1
            query_position = torch.tensor(num_processed_tokens - 1, dtype=torch.int16)

            if is_image_prefill:
                outputs = self.image_prefill(
                    input_chunk,
                    cache_pos_chunk,
                    block_tables,
                    local_block_tables,
                    query_position,
                    chunked_attention_mask,
                    position_ids_chunk,
                    lora_int_ids if self.rbln_config.use_lora else None,
                )
            else:
                outputs = self.prefill(
                    input_chunk,
                    cache_pos_chunk,
                    block_tables,
                    local_block_tables,
                    query_position,
                    chunked_attention_mask,
                    position_ids_chunk,
                    lora_int_ids if self.rbln_config.use_lora else None,
                )

            if self.rbln_config.output_hidden_states:
                output_logits.append(outputs[0])
                all_hidden_states.append(tuple(outputs[1:]))
            else:
                output_logits.append(outputs)

            padded_cache_lengths += current_padded_cache_lengths
            step += num_processed_tokens

        if self.rbln_config.output_hidden_states:
            num_hidden_layers = len(all_hidden_states[0]) - 1
            concatenated_hidden_states = ()
            for l_idx in range(num_hidden_layers + 1):
                l_hidden_states = torch.cat([hidden_states[l_idx] for hidden_states in all_hidden_states], dim=1)
                l_hidden_states = l_hidden_states[:, :query_length, :]
                concatenated_hidden_states += (l_hidden_states,)

            all_hidden_states = concatenated_hidden_states

        # Aggregate output_logits
        output_logits = torch.concat(output_logits, dim=-2)
        if self.rbln_config.logits_to_keep > 0:
            output_logits = output_logits[:, -self.rbln_config.logits_to_keep :, :]
        else:
            output_logits = output_logits[:, :query_length, :]
            # index copy for masked output_logits
            if attention_mask is not None:
                new_output_logits = torch.full(
                    (1, attention_mask.shape[-1], output_logits.shape[-1]),
                    fill_value=1e-10,
                    dtype=output_logits.dtype,
                )
                mask_indices = torch.nonzero(attention_mask, as_tuple=True)[0]
                new_output_logits.index_copy_(dim=-2, index=mask_indices, source=output_logits)

            output_logits = new_output_logits

        if not is_external_block_tables:
            self.dec_attn_mask[batch_idx : batch_idx + 1] = chunked_attention_mask

        return RBLNGemma3ForCausalLMOutput(
            logits=output_logits,
            padded_cache_lengths=padded_cache_lengths,
            attention_mask=chunked_attention_mask,
            hidden_states=all_hidden_states,
        )
