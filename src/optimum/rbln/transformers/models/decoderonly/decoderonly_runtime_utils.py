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


_PACKED_LAST_LAYER_LN_IO_COUNT = 11
_PACKED_ATTENTION_IO_COUNT = 5  # q_proj, k_proj, v_proj, attn_op_out, attn_o_proj_out
_PACKED_ATTENTION_ROT_COUNT = 2  # attn_q_rot, attn_k_rot
_PACKED_LAST_LAYER_LN_IO_EXT_COUNT = _PACKED_LAST_LAYER_LN_IO_COUNT + _PACKED_ATTENTION_IO_COUNT
_PACKED_LAST_LAYER_LN_IO_FULL_COUNT = _PACKED_LAST_LAYER_LN_IO_EXT_COUNT + _PACKED_ATTENTION_ROT_COUNT
_PER_LAYER_BASE_COUNT = 9  # pre_in, pre_out, post_in, post_out, gate, act, up, mul, down
_PER_LAYER_EXT_COUNT = _PER_LAYER_BASE_COUNT + _PACKED_ATTENTION_IO_COUNT  # + q, k, v, attn_op, attn_o_proj
_PER_LAYER_FULL_COUNT = _PER_LAYER_EXT_COUNT + _PACKED_ATTENTION_ROT_COUNT  # + q_rot, k_rot
_FINAL_NORM_COUNT = 2  # final_norm_in, final_norm_out


def _rebuild_last_layer_layernorm_io(packed: list[torch.Tensor], num_hidden_layers: int) -> dict[str, Any]:
    # packed order must match DecoderOnlyModel.forward (tracing path)
    pre_in, pre_out = packed[0], packed[1]
    post_in, post_out = packed[2], packed[3]
    gate, act, up, mul, down = packed[4], packed[5], packed[6], packed[7], packed[8]
    final_in, final_out = packed[9], packed[10]
    per_layer: list[Optional[dict[str, Any]]] = [None for _ in range(num_hidden_layers)]
    last: dict[str, Any] = {
        "pre_attention": (pre_in, pre_out),
        "post_attention": (post_in, post_out),
        "mlp_gate": gate,
        "mlp_act": act,
        "mlp_up": up,
        "mlp_mul": mul,
        "mlp_down": down,
    }
    if len(packed) >= _PACKED_LAST_LAYER_LN_IO_EXT_COUNT:
        q, k, v, attn_op_out, attn_o_proj_out = packed[11], packed[12], packed[13], packed[14], packed[15]
        last["attn_q_proj"] = q
        last["attn_k_proj"] = k
        last["attn_v_proj"] = v
        last["attn_op_out"] = attn_op_out
        last["attn_o_proj_out"] = attn_o_proj_out
    if len(packed) >= _PACKED_LAST_LAYER_LN_IO_FULL_COUNT:
        last["attn_q_rot"] = packed[16]
        last["attn_k_rot"] = packed[17]
    per_layer[-1] = last
    return {"per_layer": per_layer, "final_norm": (final_in, final_out)}


def _rebuild_all_layers_layernorm_io(packed: list[torch.Tensor], num_hidden_layers: int) -> dict[str, Any]:
    # packed order must match DecoderOnlyModel.forward (tracing path, all-layers stack)
    pre_in_s, pre_out_s = packed[0], packed[1]
    post_in_s, post_out_s = packed[2], packed[3]
    gate_s, act_s, up_s, mul_s, down_s = packed[4], packed[5], packed[6], packed[7], packed[8]
    final_in, final_out = packed[9], packed[10]
    q_s = k_s = v_s = attn_op_out_s = attn_o_proj_out_s = None
    q_rot_s = k_rot_s = None
    if len(packed) >= _PACKED_LAST_LAYER_LN_IO_EXT_COUNT:
        q_s, k_s, v_s, attn_op_out_s, attn_o_proj_out_s = packed[11], packed[12], packed[13], packed[14], packed[15]
    if len(packed) >= _PACKED_LAST_LAYER_LN_IO_FULL_COUNT:
        q_rot_s, k_rot_s = packed[16], packed[17]

    if pre_in_s.dim() < 1 or int(pre_in_s.shape[0]) != int(num_hidden_layers):
        raise ValueError(
            f"Invalid packed all-layer layernorm_io: expected leading dim L={num_hidden_layers}, got {tuple(pre_in_s.shape)}"
        )

    per_layer: list[Optional[dict[str, Any]]] = [None for _ in range(num_hidden_layers)]
    for i in range(num_hidden_layers):
        item: dict[str, Any] = {
            "pre_attention": (pre_in_s[i], pre_out_s[i]),
            "post_attention": (post_in_s[i], post_out_s[i]),
            "mlp_gate": gate_s[i],
            "mlp_act": act_s[i],
            "mlp_up": up_s[i],
            "mlp_mul": mul_s[i],
            "mlp_down": down_s[i],
        }
        if q_s is not None:
            item["attn_q_proj"] = q_s[i]
            item["attn_k_proj"] = k_s[i]
            item["attn_v_proj"] = v_s[i]
            item["attn_op_out"] = attn_op_out_s[i]
            item["attn_o_proj_out"] = attn_o_proj_out_s[i]
        if q_rot_s is not None:
            item["attn_q_rot"] = q_rot_s[i]
            item["attn_k_rot"] = k_rot_s[i]
        per_layer[i] = item
    return {"per_layer": per_layer, "final_norm": (final_in, final_out)}


def _rebuild_all_layers_flat_layernorm_io(
    packed: list[torch.Tensor], num_hidden_layers: int, per_layer_count: int
) -> dict[str, Any]:
    """Rebuild per-layer layernorm_io when the compiler unpacked torch.stack into flat 3D tensors.

    The expected layout is ``per_layer_count`` tensors per layer (in layer order),
    followed by 2 final-norm tensors (final_norm_in, final_norm_out).
    """
    per_layer: list[Optional[dict[str, Any]]] = [None for _ in range(num_hidden_layers)]
    idx = 0
    for i in range(num_hidden_layers):
        pre_in, pre_out = packed[idx], packed[idx + 1]
        post_in, post_out = packed[idx + 2], packed[idx + 3]
        gate, act, up, mul, down = packed[idx + 4], packed[idx + 5], packed[idx + 6], packed[idx + 7], packed[idx + 8]
        item: dict[str, Any] = {
            "pre_attention": (pre_in, pre_out),
            "post_attention": (post_in, post_out),
            "mlp_gate": gate,
            "mlp_act": act,
            "mlp_up": up,
            "mlp_mul": mul,
            "mlp_down": down,
        }
        if per_layer_count >= _PER_LAYER_EXT_COUNT:
            item["attn_q_proj"] = packed[idx + 9]
            item["attn_k_proj"] = packed[idx + 10]
            item["attn_v_proj"] = packed[idx + 11]
            item["attn_op_out"] = packed[idx + 12]
            item["attn_o_proj_out"] = packed[idx + 13]
        if per_layer_count >= _PER_LAYER_FULL_COUNT:
            item["attn_q_rot"] = packed[idx + 14]
            item["attn_k_rot"] = packed[idx + 15]
        per_layer[i] = item
        idx += per_layer_count
    final_in = packed[idx]
    final_out = packed[idx + 1]
    return {"per_layer": per_layer, "final_norm": (final_in, final_out)}


def _rebuild_layernorm_io(packed: list[torch.Tensor], num_hidden_layers: int) -> dict[str, Any]:
    t0 = packed[0]
    # Case 1: 4D stacked tensors (compiler preserved torch.stack)
    if hasattr(t0, "dim") and int(t0.dim()) >= 4 and int(t0.shape[0]) == int(num_hidden_layers):
        return _rebuild_all_layers_layernorm_io(packed, num_hidden_layers)
    # Case 2: flat 3D tensors — compiler unpacked the stack into individual per-layer outputs
    n_packed = len(packed)
    for plc in (_PER_LAYER_FULL_COUNT, _PER_LAYER_EXT_COUNT, _PER_LAYER_BASE_COUNT):
        expected = plc * num_hidden_layers + _FINAL_NORM_COUNT
        if n_packed == expected:
            return _rebuild_all_layers_flat_layernorm_io(packed, num_hidden_layers, per_layer_count=plc)
    # Fallback: last-layer only
    return _rebuild_last_layer_layernorm_io(packed, num_hidden_layers)


def _normalize_prefill_out_buffer(t: torch.Tensor) -> torch.Tensor:
    """Align preallocated CPU buffers with rebel ``_output_profile`` shapes.

    Some traced paths (e.g. rotary cos/sin ``unsqueeze``, stacked LN I/O) yield
    ``(1, 1, S, H)`` while the compiled graph expects ``(1, S, H)``. Drop a
    leading singleton middle dimension only in that case.
    """
    if t.dim() == 4 and int(t.shape[0]) == 1 and int(t.shape[1]) == 1:
        return t.squeeze(1).contiguous()
    return t


def _alloc_backing_for_profile_output(
    prof_shape: tuple[int, ...],
    padded_mask_length: int,
    fill_value: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate host storage for one compiled output; sequence dim follows the profile when fixed (e.g. logits ``(1,1,V)``)."""
    if len(prof_shape) == 3:
        mid = int(prof_shape[1])
        seq_len = 1 if mid == 1 else padded_mask_length
        return torch.full((1, seq_len, int(prof_shape[-1])), fill_value, dtype=dtype)
    if len(prof_shape) == 4:
        a, b, _seq, d = (int(prof_shape[i]) for i in range(4))
        return torch.full((a, b, padded_mask_length, d), fill_value, dtype=dtype)
    raise ValueError(f"Unsupported prefill output profile rank/shape: {prof_shape}")


def _chunk_view_for_profile_output(
    backing: torch.Tensor,
    prof_shape: tuple[int, ...],
    *,
    s_idx: int,
    chunk_sz: int,
) -> torch.Tensor:
    """Slice ``backing`` so it matches the compiled graph's shape for this chunk."""
    if len(prof_shape) == 3 and int(prof_shape[1]) == 1:
        return _normalize_prefill_out_buffer(backing)
    if len(prof_shape) == 3:
        return _normalize_prefill_out_buffer(backing[:, s_idx : s_idx + chunk_sz, :])
    if len(prof_shape) == 4:
        return _normalize_prefill_out_buffer(backing[:, :, s_idx : s_idx + chunk_sz, :])
    raise ValueError(f"Unsupported prefill output profile rank/shape: {prof_shape}")


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

        out_ln = bool(getattr(self.rbln_config, "output_layernorm_io", False))
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
            offset = 1
            hs = None
            if self.rbln_config.output_hidden_states:
                hs_count = int(self.config.num_hidden_layers) + 1
                hs = tuple(outputs[offset : offset + hs_count])
                offset += hs_count
            ln_io = None
            if out_ln and len(outputs) >= offset + _PACKED_LAST_LAYER_LN_IO_COUNT:
                # Newer compiled variants may append attention intermediates after the base 11 tensors.
                if len(outputs) >= offset + _PACKED_LAST_LAYER_LN_IO_FULL_COUNT:
                    take = _PACKED_LAST_LAYER_LN_IO_FULL_COUNT
                elif len(outputs) >= offset + _PACKED_LAST_LAYER_LN_IO_EXT_COUNT:
                    take = _PACKED_LAST_LAYER_LN_IO_EXT_COUNT
                else:
                    take = _PACKED_LAST_LAYER_LN_IO_COUNT
                packed = list(outputs[offset : offset + take])
                ln_io = _rebuild_layernorm_io(packed, int(self.config.num_hidden_layers))
        else:
            logits = outputs
            hs = None
            ln_io = None

        if out_ln and ln_io is None:
            # Some compiled runtime variants do not expose layernorm tensors yet.
            # Keep a non-empty marker instead of hardcoded None.
            ln_io = {"per_layer": [], "final_norm": None, "runtime_layernorm_io_unavailable": True}

        lhs = hs[-1] if hs else None
        return RBLNDecoderOnlyOutput(
            logits=logits,
            hidden_states=hs,
            layernorm_io=ln_io,
            last_hidden_state=lhs,
        )

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
        out_ln: bool = False,
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

        profile = self.runtime._output_profile
        fill = 1e-10
        dtype = self.rbln_config.dtype

        # One backing tensor per compiled output; last dims (e.g. 512 vs intermediate_size) come from the graph.
        backing: list[torch.Tensor] = [
            _alloc_backing_for_profile_output(tuple(p.shape), padded_mask_length, fill, dtype) for p in profile
        ]

        output_logits = backing[0]

        n_hs = (int(self.config.num_hidden_layers) + 1) if self.rbln_config.output_hidden_states else 0
        output_hidden_states: Optional[list[torch.Tensor]] = None
        if n_hs > 0:
            if 1 + n_hs > len(profile):
                raise RuntimeError(
                    f"Prefill output profile has only {len(profile)} outputs; "
                    f"need at least {1 + n_hs} for logits plus {n_hs} hidden state tensors."
                )
            output_hidden_states = [backing[1 + j] for j in range(n_hs)]

        ln_start = 1 + n_hs
        output_packed_ln_io: Optional[list[torch.Tensor]] = None
        if out_ln:
            if ln_start > len(profile):
                raise RuntimeError("Prefill output profile has no layernorm_io tensors after hidden states.")
            output_packed_ln_io = [backing[j] for j in range(ln_start, len(profile))]

        num_chunks = padded_input_length // self.rbln_config.prefill_chunk_size
        chunk_sz = self.rbln_config.prefill_chunk_size

        for i in range(num_chunks):
            s_idx = i * chunk_sz + valid_start_index
            chunk_out: list[torch.Tensor] = []
            for j, prof in enumerate(profile):
                shp = tuple(prof.shape)
                chunk_out.append(
                    _chunk_view_for_profile_output(
                        backing[j],
                        shp,
                        s_idx=s_idx,
                        chunk_sz=chunk_sz,
                    )
                )
            out_buffers[i] = chunk_out

        return out_buffers, output_logits, output_hidden_states, output_packed_ln_io

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

        out_ln = bool(getattr(self.rbln_config, "output_layernorm_io", False))
        out_buffers, output_logits, output_hidden_states, output_packed_ln_io = self._prepare_prefill_outputs(
            query_length, attention_mask, out_ln=out_ln
        )

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

        layernorm_io = None
        if out_ln:
            if output_packed_ln_io is None:
                layernorm_io = {"per_layer": [], "final_norm": None, "runtime_layernorm_io_unavailable": True}
            else:
                packed = [
                    (buf[:, :, :-padding_size, :] if buf.dim() >= 4 else buf[:, :-padding_size, :])
                    for buf in output_packed_ln_io
                ] if padding_size > 0 else output_packed_ln_io
                layernorm_io = _rebuild_layernorm_io(packed, int(self.config.num_hidden_layers))

        lhs = all_hidden_states[-1] if all_hidden_states else None
        return RBLNDecoderOnlyOutput(
            logits=output_logits,
            padded_cache_lengths=padded_cache_lengths,
            hidden_states=all_hidden_states,
            layernorm_io=layernorm_io,
            last_hidden_state=lhs,
        )
