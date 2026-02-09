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

from collections import deque
from typing import TYPE_CHECKING, Any, Optional

import rebel
import torch
import torch.nn.functional as F

from ....utils.runtime_utils import RBLNPytorchRuntime
from ...modeling_outputs import RBLNDecoderOnlyOutput
from .configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


if TYPE_CHECKING:
    from transformers.configuration_utils import PreTrainedConfig


class RBLNPageTableManager:
    EMPTY_BLOCK = -1
    NO_BLOCKS_ERROR = (
        "No memory blocks are available for allocation. "
        "The generate() API cannot complete this inference task because Paged Attention is not fully supported by optimum-rbln. "
        "This is supported by vllm-rbln (see: https://docs.rbln.ai/software/model_serving/vllm_support/vllm-rbln.html). "
        "Using vllm-rbln should fix this issue and enhance inference performance."
    )

    def __init__(self, rbln_config: RBLNDecoderOnlyModelForCausalLMConfig):
        self.rbln_config = rbln_config
        self.block_tables = torch.zeros(
            self.rbln_config.batch_size,
            self.rbln_config.max_seq_len // self.rbln_config.kvcache_block_size,
            dtype=torch.int16,
        ).fill_(self.EMPTY_BLOCK)
        self.free_block_pool = deque(x for x in range(self.rbln_config.kvcache_num_blocks))

    def update_block(self, batch_idx: int, block_idx: int):
        """
        If the block is empty (empty_block), allocates a block from the free_block_pool.
        """
        if batch_idx >= len(self.block_tables) or block_idx >= len(self.block_tables[batch_idx]):
            raise IndexError(
                f"Invalid index(batch_idx={batch_idx}, block_idx={block_idx}): \n \
                               BlockTable Shape(batch_axis, block_axis): {self.block_tables.shape}, BlockSize: {self.rbln_config.kvcache_block_size}"
            )

        if self.block_tables[batch_idx][block_idx] == self.EMPTY_BLOCK:
            if self.free_block_pool:
                block = self.free_block_pool.popleft()
                self.block_tables[batch_idx][block_idx] = block
            else:
                raise RuntimeError(self.NO_BLOCKS_ERROR)

    def replace_empty_block(self, block_tables: torch.Tensor):
        """
        Replaces all occurrences of `self.empty_block` in `block_tables` with a dummy block from `self.free_block_pool`.
        """
        if not torch.any(block_tables == self.EMPTY_BLOCK):
            return block_tables.clone()
        elif self.free_block_pool:
            _free_block = self.free_block_pool[0]
            return torch.where(block_tables == self.EMPTY_BLOCK, _free_block, block_tables)
        else:
            raise RuntimeError(self.NO_BLOCKS_ERROR)

    def get_block_tables(
        self, cache_position: torch.Tensor, batch_idx: int = None, batch_size: int = None, phase: str = "prefill"
    ) -> torch.Tensor:
        """
        Manages and returns the KV cache block tables.
        Updates the block tables based on the given cache_position, allocating new blocks or reusing existing ones as needed.

        Args:
            cache_position (torch.Tensor): Tensor containing cache position information, indicating positions within the cache for each batch item.
            batch_idx (int, optional): Specific batch index, used when phase is 'prefill'.

        Returns:
            Updated block tables.
        """

        def get_global_block_tables():
            if not self.rbln_config.use_global_attention:
                return None

            if phase == "prefill":
                # Track previously used blocks and return them to the free_block_pool and
                # reset the current batch's block table to empty blocks
                prev_blocks = self.block_tables[batch_idx][self.block_tables[batch_idx] != self.EMPTY_BLOCK].tolist()
                self.free_block_pool.extend(prev_blocks)
                self.block_tables[batch_idx].fill_(self.EMPTY_BLOCK)

                # Get the start (s) and end (e) positions from cache_position and
                # iterate over the cache positions to allocate necessary blocks
                s, e = cache_position[0][0].item(), cache_position[0][-1].item()
                for position in range(s, e + 1, self.rbln_config.kvcache_block_size):
                    block_idx = position // self.rbln_config.kvcache_block_size
                    self.update_block(batch_idx, block_idx)

                return self.replace_empty_block(self.block_tables[batch_idx])
            # Case for 'decoder' phase, iterate over the cache positions to allocate necessary blocks
            else:
                for b_idx in range(batch_size):
                    position = cache_position[b_idx][0].item()
                    block_idx = position // self.rbln_config.kvcache_block_size
                    self.update_block(b_idx, block_idx)

                return self.replace_empty_block(self.block_tables)

        def get_local_block_tables():
            if not self.rbln_config.use_local_attention:
                return None
            else:
                return (
                    torch.tensor([batch_idx], dtype=torch.int16)
                    if phase == "prefill"
                    else torch.arange(batch_size, dtype=torch.int16).view(batch_size, -1)
                )

        return get_global_block_tables(), get_local_block_tables()

    # Whether block_tables and local_block_tables are provided by the user
    def is_external_block_tables(
        self, block_tables: Optional[torch.Tensor], local_block_tables: Optional[torch.Tensor]
    ):
        if self.rbln_config.cache_impl == "static" and block_tables is None:
            return False
        elif self.rbln_config.cache_impl == "sliding_window" and local_block_tables is None:
            return False
        elif self.rbln_config.cache_impl == "hybrid":
            if (block_tables is not None) != (local_block_tables is not None):
                raise ValueError(
                    "Both block_tables and local_block_tables must be provided or neither of them must be provided."
                )
            elif block_tables is None and local_block_tables is None:
                return False

        return True

    def get_block_tables_if_needed(
        self,
        batch_size,
        cache_position: torch.Tensor,
        batch_idx: int = None,
        phase: str = "prefill",
        block_tables: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
    ):
        is_external_block_tables = self.is_external_block_tables(block_tables, local_block_tables)
        if not is_external_block_tables:
            block_tables, local_block_tables = self.get_block_tables(
                cache_position, batch_idx=batch_idx, batch_size=batch_size, phase=phase
            )

        return block_tables, local_block_tables, is_external_block_tables


class RBLNRuntimeModel(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name", "embed_tokens"]

    def __init__(
        self,
        runtime: rebel.Runtime,
        phase: str,
        batch_size: int,
        dec_attn_mask: torch.Tensor,
        page_table_manager: RBLNPageTableManager,
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
        config: Optional["PreTrainedConfig"] = None,
        logits_last_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(runtime, **kwargs)
        self.phase = phase
        self.batch_size = batch_size
        self.rbln_config = rbln_config
        self.config = config
        self.logits_last_dim = logits_last_dim

        # shared resources between prefill and decode phase
        self.dec_attn_mask = dec_attn_mask
        self.page_table_manager = page_table_manager
        self.out_buffers = None

        if self.phase == "prefill":
            self.causal_mask = 1 - torch.triu(
                torch.ones(1, 1, self.rbln_config.prefill_chunk_size, self.rbln_config.prefill_chunk_size), diagonal=1
            )

        self.lora_int_ids = None

    def inputs_embeddings_if_needed(
        self, input_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")

        if self.rbln_config.use_inputs_embeds:
            return self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        else:
            return input_ids

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
    ) -> torch.FloatTensor:
        if self.rbln_config.use_lora and lora_int_ids is None:
            if self.lora_int_ids is None:
                raise ValueError(
                    "lora_int_id is required when using LoRA. "
                    "You should call set_lora_int_ids() before forward() or pass lora_int_id to forward()."
                )

            lora_int_ids = self.lora_int_ids

        if lora_int_ids is not None and lora_int_ids.shape[0] != self.batch_size:
            raise ValueError(f"lora_int_ids size mismatch: got {lora_int_ids.shape[0]}, expected {self.batch_size}.")

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
                        f"Decoding step {decoding_step} out of bounds for attention mask with shape {self.dec_attn_mask.shape}."
                    )

                if self.rbln_config.use_position_ids:
                    self.dec_attn_mask[b_idx, decoding_step] = 1

                    if self.batch_size < block_tables.shape[0]:
                        block_tables = block_tables[: self.batch_size]

                    if self.dec_attn_mask is not None and self.batch_size < self.dec_attn_mask.shape[0]:
                        self.dec_attn_mask = self.dec_attn_mask[: self.batch_size]
                else:
                    if is_external_block_tables:
                        self.dec_attn_mask[b_idx].fill_(0)
                        self.dec_attn_mask[b_idx, :, :, : decoding_step + 1] = 1
                    else:
                        self.dec_attn_mask[b_idx, :, :, decoding_step] = 1

            attention_mask = self.dec_attn_mask

        outputs = super().forward(
            inputs,
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

    def _prepare_prefill_inputs(
        self,
        inputs: torch.Tensor,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embed: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        """
        Prepare inputs for prefill phase.
        """
        # Handle continuous batching in a compiled graph by extracting valid inputs
        # If an attention mask is provided, select only the valid (non-masked) inputs
        if attention_mask is not None:
            if attention_mask.dim() != 1:
                raise ValueError("attention_mask must be a 1D tensor.")

            mask_bool = attention_mask.to(dtype=torch.bool)
            if (~mask_bool).any():
                indice_one = torch.nonzero(mask_bool, as_tuple=False)
                if indice_one.numel() == 0:
                    raise ValueError("attention_mask with padding must include at least one real token.")
                first_one_idx, last_one_idx = int(indice_one[0].item()), int(indice_one[-1].item())
                if last_one_idx - first_one_idx + 1 != mask_bool.sum():
                    raise ValueError(
                        "attention_mask must group all 1s together (e.g. 000111 or 1111000). "
                        "Zeros between real tokens like 101010 are not supported."
                    )

                if self.rbln_config.can_generate and not mask_bool[first_one_idx:].all():
                    raise ValueError("attention_mask must be left padded for generation.")

            inputs = inputs[:, mask_bool]
            position_embed = None if position_embed is None else position_embed[:, :, :, mask_bool, :]
            token_type_ids = None if token_type_ids is None else token_type_ids[:, mask_bool]

        query_length = inputs.shape[1]
        if query_length > self.rbln_config.max_seq_len:
            raise ValueError(
                f"Input length ({query_length}) exceeds the maximum allowed sequence length ({self.rbln_config.max_seq_len})."
            )

        # Initialize attention mask for chunked processing
        if self.rbln_config.use_attention_mask:
            if self.rbln_config.use_position_ids:
                chunked_attention_mask = torch.zeros(1, self.rbln_config.max_seq_len, dtype=self.rbln_config.dtype)
            else:
                chunked_attention_mask = torch.zeros(
                    1,
                    1,
                    self.rbln_config.prefill_chunk_size,
                    self.rbln_config.max_seq_len,
                    dtype=self.rbln_config.dtype,
                )
        else:
            chunked_attention_mask = None

        cache_position = (
            torch.arange(query_length, dtype=torch.int32).unsqueeze(0) if cache_position is None else cache_position
        )
        # Pad input and cache_position if the last chunk is smaller than `prefill_chunk_size`
        padding_size = (self.rbln_config.prefill_chunk_size - query_length) % self.rbln_config.prefill_chunk_size
        if padding_size > 0:
            inputs = (
                F.pad(inputs, (0, 0, 0, padding_size))
                if self.rbln_config.use_inputs_embeds
                else F.pad(inputs, (0, padding_size))
            )
            position_embed = F.pad(position_embed, (0, 0, 0, padding_size)) if position_embed is not None else None
            token_type_ids = F.pad(token_type_ids, (0, padding_size), value=-1) if token_type_ids is not None else None
            cache_position = F.pad(cache_position, (0, padding_size))

        # Overwrite position_ids and padded_cache_lengths
        if self.rbln_config.use_position_ids and position_ids is None:
            position_ids = cache_position.clone()
        else:
            position_ids = position_ids

        padded_cache_lengths = 0

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

    def _prepare_prefill_outputs(
        self,
        query_length: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # Prepare out buffers
        padding_size = (self.rbln_config.prefill_chunk_size - query_length) % self.rbln_config.prefill_chunk_size
        padded_input_length = query_length + padding_size
        padded_mask_length = (
            attention_mask.shape[-1] + padding_size if attention_mask is not None else padded_input_length
        )
        out_buffers = [[] for _ in range(padded_input_length // self.rbln_config.prefill_chunk_size)]

        valid_start_index = (
            int(torch.nonzero(attention_mask, as_tuple=False)[0][0].item()) if attention_mask is not None else 0
        )

        if self.logits_last_dim is None:
            logits_last_dim = self.config.vocab_size if self.rbln_config.can_generate else self.config.hidden_size
        else:
            logits_last_dim = self.logits_last_dim

        # Prepare logits buffer
        logits_size = (
            1,
            1 if self.rbln_config.logits_to_keep == 1 else padded_mask_length,
            logits_last_dim,
        )
        output_logits = torch.full(logits_size, fill_value=1e-10, dtype=self.rbln_config.dtype)

        if self.rbln_config.logits_to_keep == 1:
            for i in range(padded_input_length // self.rbln_config.prefill_chunk_size):
                out_buffers[i].append(output_logits)
        else:
            for i in range(padded_input_length // self.rbln_config.prefill_chunk_size):
                s_idx = i * self.rbln_config.prefill_chunk_size + valid_start_index
                out_buffers[i].append(output_logits[:, s_idx : s_idx + self.rbln_config.prefill_chunk_size])

        # Prepare output hidden states
        output_hidden_states = None
        if self.rbln_config.output_hidden_states:
            hidden_states_size = (
                1,
                padded_mask_length,
                self.config.hidden_size,
            )
            output_hidden_states = [
                torch.full(hidden_states_size, fill_value=1e-10, dtype=self.rbln_config.dtype)
                for _ in range(self.config.num_hidden_layers + 1)
            ]

            for i in range(padded_input_length // self.rbln_config.prefill_chunk_size):
                s_idx = i * self.rbln_config.prefill_chunk_size + valid_start_index
                out_buffers[i].extend(
                    [
                        hidden_states_buffer[:, s_idx : s_idx + self.rbln_config.prefill_chunk_size]
                        for hidden_states_buffer in output_hidden_states
                    ]
                )

        return out_buffers, output_logits, output_hidden_states

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
            inputs, cache_position, attention_mask, position_ids, position_embed, token_type_ids=token_type_ids
        )

        out_buffers, output_logits, output_hidden_states = self._prepare_prefill_outputs(query_length, attention_mask)

        # Assumed that prefix caching was performed externally if cache_position doesn't start from 0.
        prefix_cached_len = cache_position[0][0].item()
        if prefix_cached_len > 0:
            if prefix_cached_len % self.rbln_config.prefill_chunk_size != 0:
                raise NotImplementedError(
                    "Prefix Caching is not supported yet for non-multiple of prefill_chunk_size."
                )
            if self.rbln_config.use_attention_mask:
                if self.rbln_config.use_position_ids:
                    chunked_attention_mask[:, :prefix_cached_len] = 1
                else:
                    chunked_attention_mask[:, :, :, :prefix_cached_len] = 1

        # Process input in chunks of size `prefill_chunk_size`
        for i, step in enumerate(range(0, query_length, self.rbln_config.prefill_chunk_size)):
            s, e = step, step + self.rbln_config.prefill_chunk_size
            # Extract the current chunk of inputs, cache positions, position ids, and position embeddings
            input_chunk = inputs[:, s:e]
            cache_pos_chunk = cache_position[:, s:e]
            position_ids_chunk = position_ids[:, s:e] if self.rbln_config.use_position_ids else None
            position_embed_chunk = position_embed[:, :, :, s:e, :] if position_embed is not None else None

            # Update attention mask to ensure proper causal behavior
            if self.rbln_config.use_attention_mask:
                if self.rbln_config.use_position_ids:
                    if step > 0:  # update previous chunk
                        # Update attention mask for the previous chunk (from s - prefill_chunk_size to s)
                        prev_chunk_start = s - self.rbln_config.prefill_chunk_size + prefix_cached_len
                        prev_chunk_end = s + prefix_cached_len
                        chunked_attention_mask[:, prev_chunk_start:prev_chunk_end] = 1

                    current_chunk_start = s + prefix_cached_len
                    current_chunk_end = min(e, query_length) + prefix_cached_len
                    if current_chunk_end > current_chunk_start:
                        chunked_attention_mask[:, current_chunk_start:current_chunk_end] = 1

                else:
                    if step > 0:  # update previous chunk
                        # Update attention mask for the previous chunk (from s - prefill_chunk_size to s)
                        prev_chunk_start = s - self.rbln_config.prefill_chunk_size + prefix_cached_len
                        prev_chunk_end = s + prefix_cached_len
                        chunked_attention_mask[:, :, :, prev_chunk_start:prev_chunk_end] = 1

                    current_chunk_start = s + prefix_cached_len
                    current_chunk_end = e + prefix_cached_len
                    chunked_attention_mask[:, :, :, current_chunk_start:current_chunk_end] = self.causal_mask

            # Calculate query position if needed
            if self.rbln_config.use_local_attention or self.rbln_config.logits_to_keep > 0:
                query_position = (
                    torch.tensor((query_length - 1) % self.rbln_config.prefill_chunk_size, dtype=torch.int16)
                    if e >= query_length
                    else torch.tensor(self.rbln_config.prefill_chunk_size - 1, dtype=torch.int16)
                )
            else:
                query_position = None

            # Forward pass for the current chunk
            _ = super().forward(
                input_chunk,
                cache_pos_chunk,
                block_tables,
                local_block_tables,
                position_embed_chunk,
                query_position,
                chunked_attention_mask if self.rbln_config.use_attention_mask else None,
                position_ids_chunk,
                lora_int_ids if self.rbln_config.use_lora else None,
                out=out_buffers[i],
            )

        # Aggregate output_logits
        padding_size = (self.rbln_config.prefill_chunk_size - query_length) % self.rbln_config.prefill_chunk_size
        if self.rbln_config.logits_to_keep == 1:
            output_logits = output_logits
        elif self.rbln_config.logits_to_keep > 1:
            output_logits = output_logits[:, -padding_size - self.rbln_config.logits_to_keep : -padding_size, :]
        else:
            output_logits = output_logits[:, :-padding_size, :]

        all_hidden_states = None
        if self.rbln_config.output_hidden_states:
            all_hidden_states = [
                output_hidden_state[:, :-padding_size, :] for output_hidden_state in output_hidden_states
            ]
            all_hidden_states = tuple(all_hidden_states)

        # Update decoder attention mask with processed KV-cache length from prefill phase
        if self.rbln_config.can_generate and not is_external_block_tables and self.rbln_config.use_attention_mask:
            if self.rbln_config.use_position_ids:
                self.dec_attn_mask[batch_idx : batch_idx + 1] = chunked_attention_mask
            else:
                self.dec_attn_mask[batch_idx].fill_(0)
                self.dec_attn_mask[batch_idx, :, :, :query_length] = 1

        return RBLNDecoderOnlyOutput(
            logits=output_logits, padded_cache_lengths=padded_cache_lengths, hidden_states=all_hidden_states
        )
