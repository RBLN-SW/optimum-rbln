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

from ...modeling_outputs import RBLNDecoderOnlyOutput
from ..decoderonly.decoderonly_runtime_utils import RBLNRuntimeModel


class RBLNQwen3_5RuntimeModel(RBLNRuntimeModel):
    """Runtime for the hybrid Qwen3.5 text backbone (batch_size == 1 for now).

    ``full_attention`` layers use the on-device paged KV cache (handled by the base runtime, whose
    buffers are static and never passed at call time). The ``linear_attention`` (GatedDeltaNet) layers
    instead carry two FUNCTIONAL states per layer — ``conv_state`` and ``recurrent_state`` — which are
    ordinary graph inputs/outputs, not on-device static caches. This runtime holds them on the host and
    threads them across prefill windows and decode steps:

        zero-init -> prefill window 0 -> ... -> prefill window N -> decode step 0 -> decode step 1 -> ...

    Each compiled graph call takes the current states (appended after the standard inputs, in linear-
    layer order) and returns the updated states (right after ``logits`` in the output tuple); we read
    them back into the host store for the next call. ``RBLNPytorchRuntime.forward`` drops ``None`` args,
    so the standard-arg order collapses to exactly the order the Qwen3.5 wrapper's ``prepare_forward_args``
    expects (``inputs, cache_position, block_tables, position_embed, [query_position], [attention_mask],
    [lora]``), after which the linear states follow.
    """

    def __init__(
        self,
        *args,
        linear_attention_layers: Optional[List[int]] = None,
        conv_state_shape=None,
        recurrent_state_shape=None,
        state_dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.linear_attention_layers = list(linear_attention_layers or [])
        self.conv_state_shape = tuple(conv_state_shape)
        self.recurrent_state_shape = tuple(recurrent_state_shape)
        self.state_dtype = state_dtype
        self._conv_states = {}
        self._recurrent_states = {}
        self.reset_linear_states()

    # ------------------------------------------------------------------ state store (host)
    def reset_linear_states(self):
        """Zero both states for every linear layer — called at the start of each sequence (prefill).

        Reassign per key (keep the dict OBJECT) so a store shared with the decode runtime — set up in
        RBLNQwen3_5Model.setup_runtime for prefill->decode carry — is not detached by the reset.
        """
        for i in self.linear_attention_layers:
            self._conv_states[i] = torch.zeros(self.conv_state_shape, dtype=self.state_dtype)
            self._recurrent_states[i] = torch.zeros(self.recurrent_state_shape, dtype=self.state_dtype)

    def _store_linear_states(self, outputs) -> torch.Tensor:
        """outputs == (logits, new_conv_i, new_recur_i, ...); persist the new states, return logits."""
        outputs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
        idx = 1
        for i in self.linear_attention_layers:
            self._conv_states[i] = outputs[idx]
            self._recurrent_states[i] = outputs[idx + 1]
            idx += 2
        return outputs[0]

    def _run(self, named_inputs: dict) -> torch.Tensor:
        """Add the linear states, order inputs by the runtime's OWN signature, invoke (no ``out=``),
        capture the new states.

        rebel prunes dead graph inputs, so the runtime's live inputs are a subset of the compiled
        input_info (e.g. an all-linear model uses no ``block_tables`` / ``cache_position`` /
        ``position_emb`` — those belong to full-attention layers — so they are dropped). Mapping by
        NAME via ``_index_to_input_name`` passes exactly what the runtime kept, in its index order,
        and works unchanged for the hybrid model (where nothing is pruned).
        """
        for i in self.linear_attention_layers:
            named_inputs[f"conv_state_{i}"] = self._conv_states[i]
            named_inputs[f"recurrent_state_{i}"] = self._recurrent_states[i]
        order = self.runtime._index_to_input_name
        args = [named_inputs[order[k]] for k in range(len(order))]
        return self._store_linear_states(super(RBLNRuntimeModel, self).forward(*args))

    # ------------------------------------------------------------------ prefill
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
    ) -> RBLNDecoderOnlyOutput:
        # Fresh sequence -> zero both states (window 0 reads zeros; the runtime carries thereafter).
        self.reset_linear_states()

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
        prefix_cached_len = cache_position[0][0].item()
        logits = None

        for step in range(0, inputs.shape[1], chunk):
            input_chunk = inputs[:, step : step + chunk]
            cache_pos_chunk = cache_position[:, step : step + chunk]
            position_embed_chunk = (
                position_embed[:, :, :, step : step + chunk, :] if position_embed is not None else None
            )

            # Reveal the current chunk (and previously seen tokens) in the causal attention mask.
            if self.rbln_config.use_attention_mask:
                if self.rbln_config.use_position_ids:
                    if step > 0:
                        chunked_attention_mask[:, prefix_cached_len : prefix_cached_len + step] = 1
                    cur_end = min(step + chunk, query_length) + prefix_cached_len
                    chunked_attention_mask[:, step + prefix_cached_len : cur_end] = 1
                else:
                    if step > 0:
                        chunked_attention_mask[:, :, :, prefix_cached_len : prefix_cached_len + step] = 1
                    chunked_attention_mask[
                        :, :, :, step + prefix_cached_len : step + prefix_cached_len + chunk
                    ] = self.causal_mask

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

            # For logits_to_keep == 1 every window overwrites the single logits row, so the final value
            # is the last window's (the next-token logits). Intermediate windows only advance the states.
            logits = self._run(named)

        return RBLNDecoderOnlyOutput(logits=logits, hidden_states=None)

    # ------------------------------------------------------------------ decode (seq == 1)
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
    ) -> RBLNDecoderOnlyOutput:
        if self.rbln_config.use_attention_mask and attention_mask is None:
            for b_idx in range(self.batch_size):
                decoding_step = cache_position[b_idx].item()
                if not (0 <= decoding_step < self.dec_attn_mask.shape[-1]):
                    raise ValueError(
                        f"Decoding step {decoding_step} out of bounds for attention mask "
                        f"with shape {self.dec_attn_mask.shape}."
                    )
                if self.rbln_config.use_position_ids:
                    self.dec_attn_mask[b_idx, decoding_step] = 1
                else:
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

        logits = self._run(named)
        return RBLNDecoderOnlyOutput(logits=logits, hidden_states=None)
