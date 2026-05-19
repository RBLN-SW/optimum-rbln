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

from typing import Any, Optional

import rebel
import torch
import torch.nn.functional as F

from ...modeling_outputs import RBLNDecoderOnlyOutput
from ..decoderonly.decoderonly_runtime_utils import RBLNPytorchRuntime, RBLNRuntimeModel


class RBLNGemma4RuntimeModel(RBLNRuntimeModel):
    """Runtime wrapper for the Gemma4 text decoder.

    Extends `RBLNRuntimeModel` with two Gemma4-specific responsibilities:

    1. Holds host-side references to `embed_tokens_per_layer` so the per-layer residual stream can be
       computed from raw `input_ids` before the compiled graph is invoked.
    2. Inserts the precomputed `per_layer_inputs` tensor as the second positional argument in every
       runtime call, matching the argument order of `Gemma4ForCausalLMWrapper.prepare_forward_args`.
    """

    def __init__(
        self,
        *args: Any,
        image_prefill: Optional[rebel.Runtime] = None,
        embed_tokens_per_layer: Optional[torch.nn.Module] = None,
        num_hidden_layers: Optional[int] = None,
        hidden_size_per_layer_input: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embed_tokens_per_layer = embed_tokens_per_layer
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.image_prefill = RBLNPytorchRuntime(image_prefill)
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

        chunked_attention_mask = torch.zeros(
            1, chunked_attention_mask.shape[-1], dtype=self.rbln_config.dtype
        )

        if self.rbln_config.use_image_prefill:
            padding_size = self.rbln_config.image_prefill_chunk_size
            inputs = F.pad(inputs, (0, 0, 0, padding_size))
            cache_position = F.pad(cache_position, (0, padding_size))
            position_ids = F.pad(position_ids, (0, padding_size))
            if token_type_ids is not None:
                token_type_ids = F.pad(token_type_ids, (0, padding_size), value=-1)

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

    def _partition_alignment_padding(self, abs_cache_pos: int) -> int:
        """Padding needed so that an upcoming image-prefill chunk does not cross a KV
        cache partition (flash_attn block) boundary.

        Bidirectional attention is computed within a single chunk. For flash_attn the KV
        cache is paged at `kvcache_block_size == kvcache_partition_len` granularity, and
        a single image chunk's K/V must stay inside one block so the partition mechanism
        can read it back consistently. If the image would straddle a boundary, we skip
        the remaining slots of the current partition (recorded in `padded_cache_lengths`)
        and start the image at the next partition.

        Returns the number of dummy slots to insert. Always 0 for eager attention.
        """
        if self.rbln_config.attn_impl != "flash_attn":
            return 0
        partition_len = self.rbln_config.kvcache_partition_len
        if partition_len is None:
            return 0
        image_chunk = self.rbln_config.image_prefill_chunk_size
        if image_chunk is None or image_chunk <= 0:
            return 0
        index_in_part = abs_cache_pos % partition_len
        remaining_in_part = partition_len - index_in_part
        if remaining_in_part < image_chunk:
            return remaining_in_part
        return 0

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
        per_layer_inputs: Optional[torch.Tensor] = None,
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

        if per_layer_inputs is not None and attention_mask is not None:
            mask_bool = attention_mask.to(dtype=torch.bool)
            if (~mask_bool).any():
                per_layer_inputs = per_layer_inputs[:, mask_bool]

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
            inputs, cache_position, attention_mask, position_ids, position_embed, token_type_ids=token_type_ids
        )

        if per_layer_inputs is not None:
            unpadded_len = per_layer_inputs.shape[1]
            target_len = inputs.shape[1]
            pad = target_len - unpadded_len
            if pad > 0:
                per_layer_inputs = F.pad(per_layer_inputs, (0, 0, 0, 0, 0, pad))
            elif pad < 0:
                per_layer_inputs = per_layer_inputs[:, :target_len]

        step = 0
        output_logits = []
        all_hidden_states_list = [] if self.rbln_config.output_hidden_states else None

        while step < query_length:
            if self.rbln_config.use_image_prefill and token_type_ids is not None:
                image_chunk = self.rbln_config.image_prefill_chunk_size
                is_image_prefill = bool(
                    torch.all(token_type_ids[:, step : step + image_chunk] == 1)
                )
                is_text_prefill_with_image_tokens = (
                    not is_image_prefill
                    and bool(
                        torch.any(
                            token_type_ids[:, step : step + self.rbln_config.prefill_chunk_size] == 1
                        )
                    )
                )
            else:
                is_image_prefill, is_text_prefill_with_image_tokens = False, False

            if is_image_prefill:
                abs_cache_pos = int(cache_position[0, step].item()) + int(padded_cache_lengths)
                extra_pad = self._partition_alignment_padding(abs_cache_pos)
                if extra_pad > 0:
                    padded_cache_lengths += extra_pad
                    continue

            is_last_chunk = step + self.rbln_config.prefill_chunk_size >= query_length

            input_chunk = inputs[:, step : step + self.rbln_config.prefill_chunk_size]
            per_layer_chunk = (
                per_layer_inputs[:, step : step + self.rbln_config.prefill_chunk_size]
                if per_layer_inputs is not None
                else None
            )
            cache_pos_chunk = (
                cache_position[:, step : step + self.rbln_config.prefill_chunk_size]
                + padded_cache_lengths
            )
            position_ids_chunk = (
                position_ids[:, step : step + self.rbln_config.prefill_chunk_size]
                if self.rbln_config.use_position_ids
                else None
            )

            num_processed_tokens = self.rbln_config.prefill_chunk_size
            current_padded_cache_lengths = 0
            if is_text_prefill_with_image_tokens:
                first_image_token_idx = torch.where(
                    token_type_ids[:, step : step + self.rbln_config.prefill_chunk_size] == 1
                )[1][0]
                num_processed_tokens = int(first_image_token_idx.item())
                current_padded_cache_lengths = (
                    self.rbln_config.prefill_chunk_size - num_processed_tokens
                )
            if is_last_chunk:
                num_processed_tokens = query_length - step
                current_padded_cache_lengths = 0

            chunked_attention_mask[
                :, step + padded_cache_lengths : step + num_processed_tokens + padded_cache_lengths
            ] = 1
            query_position = torch.tensor(num_processed_tokens - 1, dtype=torch.int16)

            runtime = self.image_prefill if is_image_prefill else self.prefill
            outputs = runtime(
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

            if self.rbln_config.output_hidden_states:
                output_logits.append(outputs[0])
                all_hidden_states_list.append(tuple(outputs[1:]))
            else:
                output_logits.append(outputs)

            padded_cache_lengths += current_padded_cache_lengths
            step += num_processed_tokens

        # Aggregate hidden states (per layer) across chunks if requested.
        if self.rbln_config.output_hidden_states:
            num_hidden_layers = len(all_hidden_states_list[0]) - 1
            concatenated_hidden_states = ()
            for l_idx in range(num_hidden_layers + 1):
                l_hidden_states = torch.cat(
                    [hs[l_idx] for hs in all_hidden_states_list], dim=1
                )[:, :query_length, :]
                concatenated_hidden_states += (l_hidden_states,)
            output_hidden_states = concatenated_hidden_states
        else:
            output_hidden_states = None

        output_logits = torch.concat(output_logits, dim=-2)
        if self.rbln_config.logits_to_keep > 0:
            output_logits = output_logits[:, -self.rbln_config.logits_to_keep :, :]
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

        if not is_external_block_tables:
            self.dec_attn_mask[batch_idx : batch_idx + 1] = chunked_attention_mask

        return RBLNDecoderOnlyOutput(
            logits=output_logits,
            padded_cache_lengths=padded_cache_lengths,
            hidden_states=output_hidden_states,
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
