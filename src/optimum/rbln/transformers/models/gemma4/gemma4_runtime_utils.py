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

from ...modeling_outputs import RBLNDecoderOnlyOutput, RBLNGemma4ForCausalLMOutput
from ..decoderonly.decoderonly_runtime_utils import RBLNPytorchRuntime
from ..decoderonly.modeling_decoderonly import RBLNRuntimeModel


def _run_length_from(tt: torch.Tensor, start: int, value: int, cap: int) -> int:
    """Length of the run of `value` in `tt[0, start:]`, capped at `cap`."""
    seg = tt[0, start : start + cap]
    diff_idx = (seg != value).nonzero(as_tuple=False)
    return int(diff_idx[0].item()) if diff_idx.numel() > 0 else int(seg.shape[0])


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

        chunked_attention_mask = torch.zeros(1, chunked_attention_mask.shape[-1], dtype=self.rbln_config.dtype)

        if position_ids is not None and position_ids.shape[-1] < cache_position.shape[-1]:
            position_ids = torch.nn.functional.pad(
                position_ids, (0, cache_position.shape[-1] - position_ids.shape[-1])
            )

        if self.rbln_config.use_image_prefill:
            padding_size = self.rbln_config.image_prefill_chunk_size
            inputs = torch.nn.functional.pad(inputs, (0, 0, 0, padding_size))
            cache_position = torch.nn.functional.pad(cache_position, (0, padding_size))
            if position_ids is not None:
                position_ids = torch.nn.functional.pad(position_ids, (0, padding_size))
            if token_type_ids is not None:
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
                if mask_bool.dim() == 2:
                    mask_bool = mask_bool[0]
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
                per_layer_inputs = torch.nn.functional.pad(per_layer_inputs, (0, 0, 0, 0, 0, pad))
            elif pad < 0:
                per_layer_inputs = per_layer_inputs[:, :target_len]

        step = 0
        minus_run_total = 0
        output_logits = []
        all_hidden_states = [] if self.rbln_config.output_hidden_states else None
        use_tt = self.rbln_config.use_image_prefill and token_type_ids is not None
        while step + minus_run_total < query_length:
            step_padded = step + minus_run_total

            if use_tt:
                tt_at = int(token_type_ids[0, step_padded].item())
                if tt_at == -1:
                    rest = token_type_ids[0, step_padded:query_length]
                    valid_off = (rest != -1).nonzero(as_tuple=True)[0]
                    run_minus = (
                        int(valid_off[0].item()) if valid_off.numel() > 0 else int(rest.shape[0])
                    )
                    minus_run_total += run_minus
                    continue
                is_image_prefill = tt_at == 1
            else:
                is_image_prefill = False

            chunk_size_used = (
                self.rbln_config.image_prefill_chunk_size if is_image_prefill else self.rbln_config.prefill_chunk_size
            )
            if use_tt:
                target_value = 1 if is_image_prefill else 0
                run_len = _run_length_from(
                    token_type_ids, step_padded, value=target_value, cap=chunk_size_used
                )
            else:
                run_len = chunk_size_used

            is_last_chunk = step_padded + run_len >= query_length

            input_chunk = inputs[:, step_padded : step_padded + chunk_size_used]
            per_layer_chunk = (
                per_layer_inputs[:, step_padded : step_padded + chunk_size_used]
                if per_layer_inputs is not None
                else None
            )
            cache_pos_chunk = cache_position[:, step : step + chunk_size_used] + padded_cache_lengths
            position_ids_chunk = (
                position_ids[:, step_padded : step_padded + chunk_size_used]
                if self.rbln_config.use_position_ids
                else None
            )

            num_processed_tokens = run_len
            current_padded_cache_lengths = chunk_size_used - run_len
            if is_last_chunk:
                tail = query_length - step_padded
                num_processed_tokens = min(tail, run_len) if run_len > 0 else tail
                current_padded_cache_lengths = 0

            chunked_attention_mask[
                :, step + padded_cache_lengths : step + num_processed_tokens + padded_cache_lengths
            ] = 1
            query_position = torch.tensor(num_processed_tokens - 1, dtype=torch.int16)

            if is_image_prefill:
                outputs = self.image_prefill(
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
            else:
                outputs = self.prefill(
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
                all_hidden_states.append(tuple(h[:, :num_processed_tokens] for h in outputs[1:]))
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

        return RBLNGemma4ForCausalLMOutput(
            logits=output_logits,
            padded_cache_lengths=padded_cache_lengths,
            attention_mask=chunked_attention_mask,
            hidden_states=all_hidden_states,
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
            tt = token_type_ids[0]
            trans = (tt[1:] != tt[:-1]).nonzero(as_tuple=True)[0] + 1
            bounds = torch.cat([trans.new_zeros(1), trans, trans.new_tensor([tt.shape[0]])])

            starts = bounds[:-1]
            run_lens = bounds[1:] - starts
            run_vals = tt[starts]
            prefill_cs = self.rbln_config.prefill_chunk_size
            image_cs = self.rbln_config.image_prefill_chunk_size
            cs_per_run = torch.where(run_vals == 1, image_cs, prefill_cs)
            valid_mask = (run_vals != -1).to(cs_per_run.dtype)
            per_run_pad = ((cs_per_run - run_lens % cs_per_run) % cs_per_run) * valid_mask
            pad_bound = int(per_run_pad.sum().item())
            if pad_bound > 0:
                extra = cache_position.new_tensor([[cache_position[0, -1].item() + pad_bound]])
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
