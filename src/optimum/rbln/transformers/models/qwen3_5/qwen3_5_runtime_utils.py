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


import torch

from ...modeling_outputs import RBLNDecoderOnlyOutput
from ..decoderonly.decoderonly_runtime_utils import RBLNRuntimeModel


class RBLNQwen3_5RuntimeModel(RBLNRuntimeModel):
    """Runtime for the hybrid Qwen3.5 text backbone (batch_size == 1 for now).

    ``full_attention`` layers use the on-device paged KV cache (handled by the base runtime, whose
    buffers are static and never passed at call time). The ``linear_attention`` (GatedDeltaNet) layers
    carry two extra states per layer — ``conv_state`` and ``recurrent_state`` — which are ALSO on-device
    STATIC caches: they are marked static (``mark_static_address``) in the Qwen3.5 compile context
    (``_qwen3_5_build_compile_context``) and read + written entirely in-graph via ``rbln_cache_update``.
    So, like the KV cache, they live in device DRAM and are NEVER passed at call time — this runtime does
    NOT hold state values on the host:

        prefill window 0 -> ... -> prefill window N -> decode step 0 -> decode step 1 -> ...

    The runtime's only linear-state job is to inject two 0/1 control masks per call — ``conv_state_mask``
    and ``recurrent_state_mask`` — which the GatedDeltaNet multiplies into the state it reads: ZEROS on
    prefill window 0 (fresh sequence, so the stale static cache is discarded) and ONES afterwards (carry
    whatever the previous window/step wrote). ``_run`` maps the named inputs onto the runtime's own
    input order via ``_index_to_input_name``.
    """

    def __init__(
        self,
        *args,
        conv_state_shape=None,
        recurrent_state_shape=None,
        state_dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.conv_state_shape = tuple(conv_state_shape)
        self.recurrent_state_shape = tuple(recurrent_state_shape)
        self.state_dtype = state_dtype

        # Prefill state masks: zeros on window 0 (fresh sequence), ones afterwards (carry over).
        self._conv_mask_zeros = torch.zeros(self.conv_state_shape, dtype=state_dtype)
        self._conv_mask_ones = torch.ones(self.conv_state_shape, dtype=state_dtype)
        self._recurrent_mask_zeros = torch.zeros(self.recurrent_state_shape, dtype=state_dtype)
        self._recurrent_mask_ones = torch.ones(self.recurrent_state_shape, dtype=state_dtype)
        self._valid_mask_prefill_full = torch.ones(1, self.rbln_config.prefill_chunk_size, 1, dtype=state_dtype)

    def _run(self, named_inputs: dict):
        """Order inputs by the runtime's own signature and invoke; return (logits, hidden_states).

        When output_hidden_states is set, the trailing `num_hidden_layers + 1` outputs are the per-layer
        hidden states — taking the LAST n_hidden avoids having to count the new_states.
        """
        order = self.runtime._index_to_input_name
        args = [named_inputs[order[k]] for k in range(len(order))]
        out = super(RBLNRuntimeModel, self).forward(*args)
        hidden_states = None
        if self.rbln_config.output_hidden_states:
            n_hidden = self.config.num_hidden_layers + 1
            hidden_states = tuple(out[-n_hidden:])
        return out[0], hidden_states

    def prefill_forward(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        batch_idx: int | None = None,
        block_tables: torch.Tensor | None = None,
        is_external_block_tables: bool | None = None,
        position_ids: torch.Tensor | None = None,
        position_embed: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        local_block_tables: torch.Tensor | None = None,
        lora_int_ids: torch.Tensor | None = None,
    ) -> RBLNDecoderOnlyOutput:
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
            inputs, cache_position, attention_mask, position_ids, position_embed, token_type_ids=token_type_ids
        )

        chunk = self.rbln_config.prefill_chunk_size
        # NOTE: Prefix caching is not supported yet: carrying the linear (GatedDeltaNet) conv/recurrent state
        # across a cached prefix isn't implemented.
        prefix_cached_len = cache_position[0][0].item()
        if prefix_cached_len > 0:
            raise NotImplementedError("Prefix caching is not supported for the Qwen3.5 hybrid model.")
        logits = None
        # For logits_to_keep == 0 (bare text model) the graph emits full-chunk hidden states as
        # "logits", so every window must be collected — keeping only the last window would drop
        # all earlier windows of a multi-chunk prompt and return padded chunk width.
        collect_full_logits = self.rbln_config.logits_to_keep == 0
        all_logits = [] if collect_full_logits else None
        all_hidden_states = [] if self.rbln_config.output_hidden_states else None

        for step in range(0, inputs.shape[1], chunk):
            input_chunk = inputs[:, step : step + chunk]
            cache_pos_chunk = cache_position[:, step : step + chunk]
            position_embed_chunk = (
                position_embed[:, :, :, step : step + chunk, :] if position_embed is not None else None
            )

            # Reveal the current chunk (and previously seen tokens) in the causal attention mask.
            if self.rbln_config.use_attention_mask:
                if step > 0:
                    chunked_attention_mask[:, :, :, prefix_cached_len : prefix_cached_len + step] = 1
                chunked_attention_mask[:, :, :, step + prefix_cached_len : step + prefix_cached_len + chunk] = (
                    self.causal_mask
                )

            query_position = (
                torch.tensor(
                    (query_length - 1) % chunk if step + chunk >= query_length else chunk - 1, dtype=torch.int16
                )
                if self.rbln_config.logits_to_keep > 0
                else None
            )

            named = {"inputs_embeds" if self.rbln_config.use_inputs_embeds else "input_ids": input_chunk}
            named["cache_position"] = cache_pos_chunk
            if block_tables is not None:
                named["block_tables"] = block_tables
            if position_embed_chunk is not None:
                named["position_emb"] = position_embed_chunk
            if self.rbln_config.logits_to_keep > 0:
                named["query_position"] = query_position
            if self.rbln_config.use_attention_mask:
                named["attention_mask"] = chunked_attention_mask
            if self.rbln_config.use_lora:
                named["lora_int_ids"] = lora_int_ids

            # State masks: ZERO the carried state on the FIRST prefill window (fresh sequence -> no prior context)
            named["conv_state_mask"] = self._conv_mask_zeros if step == 0 else self._conv_mask_ones
            named["recurrent_state_mask"] = self._recurrent_mask_zeros if step == 0 else self._recurrent_mask_ones

            # Which max-batch slot of the linear state caches this per-item (batch=1) prefill reads/writes.
            if batch_idx is not None:
                named["batch_idx"] = torch.tensor(batch_idx, dtype=torch.int16)

            # Per-token validity (1=real, 0=right-padding); the GatedDeltaNet multiplies it into g/beta to drop
            # padding. Full windows reuse the precomputed all-ones; the partial last window is built on demand.
            valid_count = max(0, min(chunk, query_length - step))
            if valid_count >= chunk:
                valid_mask = self._valid_mask_prefill_full
            else:
                valid_mask = torch.zeros(input_chunk.shape[0], chunk, 1, dtype=self.state_dtype)
                valid_mask[:, :valid_count] = 1.0
            named["valid_mask"] = valid_mask

            # For logits_to_keep == 1 every window overwrites the single logits row, so the final value
            # is the last window's (the next-token logits). Intermediate windows only advance the states.
            logits, hidden = self._run(named)
            if collect_full_logits:
                # keep only this window's valid (non-right-padding) tokens
                all_logits.append(logits[:, :valid_count, :])
            if self.rbln_config.output_hidden_states:
                # keep only this window's valid (non-right-padding) tokens
                all_hidden_states.append(tuple(h[:, :valid_count, :] for h in hidden))

        def place_at_mask_slots(valid_output: torch.Tensor) -> torch.Tensor:
            # `_prepare_prefill_inputs` strips padding (inputs[:, mask_bool]) before the graph, so the
            # graph only produced `query_length` valid tokens. Scatter them back to their mask slots
            # (padding stays zero) so outputs line up with the attention mask.
            if attention_mask is None:
                return valid_output
            full_len = attention_mask.shape[-1]
            start = int(torch.nonzero(attention_mask.reshape(-1), as_tuple=False)[0].item())
            buf = torch.zeros(1, full_len, valid_output.shape[-1], dtype=valid_output.dtype)
            buf[:, start : start + query_length, :] = valid_output
            return buf

        if collect_full_logits:
            # Concat per-window valid tokens along seq -> [1, query_length, hidden].
            logits = place_at_mask_slots(torch.cat(all_logits, dim=1))

        final_hidden_states = None
        if self.rbln_config.output_hidden_states:
            # Concat each layer's per-window valid tokens along seq -> [1, query_length, hidden].
            n_hidden = len(all_hidden_states[0])
            final_hidden_states = tuple(
                place_at_mask_slots(torch.cat([window[layer] for window in all_hidden_states], dim=1))
                for layer in range(n_hidden)
            )

        # padded_cache_lengths (from _prepare_prefill_inputs) is threaded back so the base
        # RBLNDecoderOnlyModelForCausalLM.forward can accumulate it per batch (the VL forward ignores it).
        return RBLNDecoderOnlyOutput(
            logits=logits, padded_cache_lengths=padded_cache_lengths, hidden_states=final_hidden_states
        )

    def decode_forward(
        self,
        inputs: torch.Tensor,
        cache_position: torch.Tensor = None,
        block_tables: torch.Tensor = None,
        is_external_block_tables: bool = None,
        attention_mask: torch.Tensor | None = None,
        position_embed: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        local_block_tables: torch.Tensor | None = None,
        lora_int_ids: torch.Tensor | None = None,
    ) -> RBLNDecoderOnlyOutput:
        if self.rbln_config.use_lora and lora_int_ids is None:
            if self.lora_int_ids is None:
                raise ValueError(
                    "lora_int_id is required when using LoRA. "
                    "You should call set_lora_int_ids() before forward() or pass lora_int_id to forward()."
                )
            lora_int_ids = self.lora_int_ids
        if lora_int_ids is not None and lora_int_ids.shape[0] != self.batch_size:
            raise ValueError(f"lora_int_ids size mismatch: got {lora_int_ids.shape[0]}, expected {self.batch_size}.")

        if self.rbln_config.use_attention_mask and attention_mask is None:
            for b_idx in range(self.batch_size):
                decoding_step = cache_position[b_idx].item()
                if not (0 <= decoding_step < self.dec_attn_mask.shape[-1]):
                    raise ValueError(
                        f"Decoding step {decoding_step} out of bounds for attention mask "
                        f"with shape {self.dec_attn_mask.shape}."
                    )
                self.dec_attn_mask[b_idx, :, :, decoding_step] = 1
            attention_mask = self.dec_attn_mask

        named = {"inputs_embeds" if self.rbln_config.use_inputs_embeds else "input_ids": inputs}
        named["cache_position"] = cache_position
        if block_tables is not None:
            named["block_tables"] = block_tables
        if position_embed is not None:
            named["position_emb"] = position_embed
        if self.rbln_config.use_attention_mask:
            named["attention_mask"] = attention_mask
        if self.rbln_config.use_lora:
            named["lora_int_ids"] = lora_int_ids

        logits, hidden_states = self._run(named)
        return RBLNDecoderOnlyOutput(logits=logits, hidden_states=hidden_states)
