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
    (rebel-pruned) input order via ``_index_to_input_name``; the static caches are absent from that order,
    and the graph's cache-update outputs alias the static addresses (in-place device writes), so only
    ``logits`` is kept.

    NOTE — multi-window prefill needs FLASH attention. A partial last prefill window
    (``seq_len % prefill_chunk_size != 0``) attending across PRIOR KV blocks (i.e. the 2nd or later chunk)
    is mis-lowered by the EAGER attention op (prefill logit pearson ~0.99 vs HF eager; the argmax can flip).
    FLASH attention (``max_seq_len >= 4096``, ``kvcache_partition_len >= 4096``) computes it correctly
    (~0.9999). The GatedDeltaNet / linear-state path is accurate under both. Prefer flash for Qwen3.5 hybrid.
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

    def _run(self, named_inputs: dict) -> torch.Tensor:
        """Order inputs by the runtime's OWN (pruned) signature and invoke; return logits.

        conv_state/recurrent_state are STATIC on-device caches (mark_static_address in the compile
        context) — read AND written in-graph via rbln_cache_update, so they are NOT passed at call
        time and are absent from ``_index_to_input_name``. Only the standard inputs + the 0/1 state
        masks flow through ``named_inputs``; rebel prunes dead inputs, so mapping by NAME passes exactly
        what the runtime kept, in its index order. The graph's state-cache-update outputs alias the
        static addresses (in-place device writes), so we keep only ``logits``.
        """
        order = self.runtime._index_to_input_name
        args = [named_inputs[order[k]] for k in range(len(order))]
        out = super(RBLNRuntimeModel, self).forward(*args)
        return out[0] if isinstance(out, (list, tuple)) else out

    # ------------------------------------------------------------------ prefill
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
        # Fresh sequence: no host reset needed — the static cache may hold stale DRAM, but the first
        # prefill window's conv/recurrent mask (0) zeros the read, so it starts fresh regardless.
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

            # State masks: ZERO the carried state on the FIRST prefill window (fresh sequence -> no prior
            # context) and pass it through (ones) on later windows. Same shape as one layer's states.
            fill = torch.zeros if step == 0 else torch.ones
            named["conv_state_mask"] = fill(self.conv_state_shape, dtype=self.state_dtype)
            named["recurrent_state_mask"] = fill(self.recurrent_state_shape, dtype=self.state_dtype)

            # which max-batch slot of the linear state caches this per-item (batch=1) prefill reads/writes.
            if batch_idx is not None:
                named["batch_idx"] = torch.tensor(batch_idx, dtype=torch.int16)

            # Per-token validity of THIS chunk: 1 for the real tokens, 0 for the right-padding. Built from
            # query_length (the SAME source as query_position above), NOT from the embeddings. The
            # GatedDeltaNet multiplies it into g/beta to drop padding from the recurrent-state sum / decay
            # and the conv_state extraction. Full windows -> all ones; the last (partial) window -> ones
            # for the first (query_length - step) columns, then zeros.
            valid_count = max(0, min(chunk, query_length - step))
            valid_mask = torch.zeros(input_chunk.shape[0], chunk, 1, dtype=self.state_dtype)
            valid_mask[:, :valid_count] = 1.0
            named["valid_mask"] = valid_mask

            # For logits_to_keep == 1 every window overwrites the single logits row, so the final value
            # is the last window's (the next-token logits). Intermediate windows only advance the states.
            logits = self._run(named)

        # padded_cache_lengths (from _prepare_prefill_inputs) is threaded back so the base
        # RBLNDecoderOnlyModelForCausalLM.forward can accumulate it per batch (the VL forward ignores it).
        return RBLNDecoderOnlyOutput(logits=logits, padded_cache_lengths=padded_cache_lengths, hidden_states=None)

    # ------------------------------------------------------------------ decode (seq == 1)
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

        # Decode always continues from the real carried state -> ones (no-op). The masks are gated on the
        # prefill phase in the GatedDeltaNet, so they are pruned from the decode graph and these entries
        # are simply ignored by the name-based input mapping; passed for safety if they survive.
        named["conv_state_mask"] = torch.ones(self.conv_state_shape, dtype=self.state_dtype)
        named["recurrent_state_mask"] = torch.ones(self.recurrent_state_shape, dtype=self.state_dtype)
        # Decode is seq=1 (always valid) and uses the recurrent rule, which ignores valid_mask -> pruned from
        # the decode graph; passed (all ones) only for safety if it survives, mirroring the state masks.
        named["valid_mask"] = torch.ones(inputs.shape[0], 1, 1, dtype=self.state_dtype)

        logits = self._run(named)
        return RBLNDecoderOnlyOutput(logits=logits, hidden_states=None)
