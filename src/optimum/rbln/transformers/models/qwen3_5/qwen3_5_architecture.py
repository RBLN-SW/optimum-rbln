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

import copy

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
    RotaryEmbedding,
    apply_rotary_pos_emb_partial,
    rotate_half,
    slice_and_unsqueeze_cos_sin,
)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    # Mirrors HF `apply_rotary_pos_emb_vision`: rotate in fp32 and round once.
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.float(), sin.float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Qwen3_5VisionAttention(nn.Module):
    def __init__(self, model: nn.Module, rbln_config) -> None:
        super().__init__()
        self._origin_model = model
        self.rbln_config = rbln_config
        self.num_heads = model.num_heads
        self.head_dim = getattr(model, "head_dim", model.proj.in_features // model.num_heads)
        self.qkv = model.qkv
        self.proj = model.proj
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        hidden_states = hidden_states.unsqueeze(0)
        q, k, v = (
            self.qkv(hidden_states).reshape(1, seq_length, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        )
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        attn_output = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(1, seq_length, -1)
        attn_output = self.proj(attn_output).squeeze(0)
        return attn_output


class Qwen3_5VisionBlock(nn.Module):
    def __init__(self, model: nn.Module, rbln_config) -> None:
        super().__init__()
        self._origin_model = model
        self.rbln_config = rbln_config
        self.norm1 = model.norm1
        self.norm2 = model.norm2
        self.attn = Qwen3_5VisionAttention(model.attn, rbln_config)
        self.mlp = model.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attn_mask, position_embeddings)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3_5VisionModelWrapper(nn.Module):
    """Qwen3.5 vision encoder for RBLN: transformer blocks + merger."""

    def __init__(self, model: nn.Module, rbln_config):
        super().__init__()
        self.merger = model.merger
        self.rbln_config = rbln_config
        self.blocks = nn.ModuleList([Qwen3_5VisionBlock(block, rbln_config) for block in model.blocks])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = (1.0 - attn_mask) * torch.finfo(hidden_states.dtype).min
        for block in self.blocks:
            hidden_states = block(hidden_states, attn_mask, (cos, sin))
        return self.merger(hidden_states)


class Qwen3_5GatedDeltaNet(nn.Module):
    """GatedDeltaNet token mixer for RBLN (conv_state + recurrent_state are on-device static caches).

    The delta rule runs as one ``rbln_custom_ops::gdn_prefill`` / ``gdn_decode`` op, which updates its rows
    of the recurrent cache in place like paged attention's KV cache. Everything around it keeps the Q/K head
    axis outermost so tensor parallelism can shard by it: the depthwise conv becomes K per-channel taps over
    a channel-last window, and the conv cache is stored as ``(B, Hk*(2+r)*(K-1), Dk)`` (Q, K and the r value
    heads of each Q/K head together).

    PREFILL takes 128-token chunks and DECODE one token; both hand the same recurrent cache rows on, so a
    chunk-prefill flows into recurrent decode without a reshape.
    """

    def __init__(self, linear_attn: nn.Module, rbln_config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self._phase = "prefill"

        self.in_proj_z = linear_attn.in_proj_z
        self.in_proj_b = linear_attn.in_proj_b
        self.in_proj_a = linear_attn.in_proj_a
        self.conv1d = linear_attn.conv1d
        self.norm = linear_attn.norm
        self.out_proj = linear_attn.out_proj
        self.A_log = linear_attn.A_log
        self.dt_bias = linear_attn.dt_bias

        self.key_dim = linear_attn.key_dim
        self.value_dim = linear_attn.value_dim
        self.head_k_dim = linear_attn.head_k_dim
        self.head_v_dim = linear_attn.head_v_dim
        self.num_k_heads = linear_attn.num_k_heads
        self.num_v_heads = linear_attn.num_v_heads
        self.conv_dim = linear_attn.conv_dim
        self.conv_kernel_size = linear_attn.conv_kernel_size

        # Pre-split the fused in_proj_qkv into separate Q/K/V projections: a depthwise conv is per-channel,
        # so the channel-axis split commutes with it.
        _qkv = linear_attn.in_proj_qkv
        _hidden = _qkv.weight.shape[1]
        _has_bias = _qkv.bias is not None
        _splits = [self.key_dim, self.key_dim, self.value_dim]
        _w = _qkv.weight.data.split(_splits, dim=0)
        _b = _qkv.bias.data.split(_splits, dim=0) if _has_bias else (None, None, None)
        self.in_proj_q = nn.Linear(_hidden, self.key_dim, bias=_has_bias)
        self.in_proj_k = nn.Linear(_hidden, self.key_dim, bias=_has_bias)
        self.in_proj_v = nn.Linear(_hidden, self.value_dim, bias=_has_bias)
        for _lin, _wi, _bi in zip((self.in_proj_q, self.in_proj_k, self.in_proj_v), _w, _b, strict=True):
            _lin.weight = nn.Parameter(_wi.contiguous())
            if _has_bias:
                _lin.bias = nn.Parameter(_bi.contiguous())

        self.prefill_chunk_size = rbln_config.prefill_chunk_size
        self.chunk_size = rbln_config.gdn_chunk_size

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase: str):
        self._phase = phase

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        query_position: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        slot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefill = "prefill" in self._phase and valid_mask is not None
        # One value per cache row gates both caches: the conv window broadcasts it, and the op takes it as
        # `has_initial_state`.
        if prefill and state_mask is not None:
            conv_state = conv_state * state_mask.reshape(-1, 1, 1)
        carry = state_mask if prefill else None
        # Returns no new recurrent state: the op writes it into `recurrent_state` at `slot`.
        batch_size, seq_len, _ = hidden_states.shape
        heads, ratio = self.num_k_heads, self.num_v_heads // self.num_k_heads
        dim, k_1 = self.head_k_dim, self.conv_kernel_size - 1

        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, heads, ratio, dim).permute(0, 2, 3, 1, 4)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        q_in = self.in_proj_q(hidden_states).reshape(batch_size, seq_len, heads, dim).transpose(1, 2)
        k_in = self.in_proj_k(hidden_states).reshape(batch_size, seq_len, heads, dim).transpose(1, 2)
        v_in = self.in_proj_v(hidden_states).reshape(batch_size, seq_len, heads, ratio, dim).permute(0, 2, 3, 1, 4)

        # Q, K and the r value heads of each Q/K head share one channel-last window [B, Hk*(2+r), K-1+S, D]: the
        # carried K-1 inputs, then this call's S inputs. One set of conv ops covers all of them.
        groups = heads * (2 + ratio)
        prev = conv_state.reshape(batch_size, groups, k_1, dim)
        new = torch.cat([q_in.unsqueeze(2), k_in.unsqueeze(2), v_in], dim=2).reshape(batch_size, groups, seq_len, dim)
        window = torch.cat([prev, new], dim=2)

        # The next call carries the last K-1 inputs; a right-padded prefill window ends at query_position.
        if prefill:
            position = query_position.to(torch.int).unsqueeze(0)
            carried = torch.cat([window[:, :, position + i] for i in range(1, k_1 + 1)], dim=2)
        else:
            carried = window.narrow(2, seq_len, k_1)
        new_conv_state = carried.reshape(batch_size, -1, dim)

        # Depthwise conv as K per-channel taps along the window's time axis.
        def per_group(channels):
            q_part, k_part, v_part = channels.split((self.key_dim, self.key_dim, self.value_dim))
            parts = (q_part.reshape(heads, 1, -1), k_part.reshape(heads, 1, -1), v_part.reshape(heads, ratio, -1))
            return torch.cat(parts, dim=1).reshape(groups, dim, *channels.shape[1:])

        taps = per_group(self.conv1d.weight[:, 0])
        out = sum(window.narrow(2, tap, seq_len) * taps[..., tap].unsqueeze(-2) for tap in range(k_1 + 1))
        if self.conv1d.bias is not None:
            out = out + per_group(self.conv1d.bias).unsqueeze(-2)
        convolved = F.silu(out).reshape(batch_size, heads, 2 + ratio, seq_len, dim)
        query, key, value = convolved[:, :, 0], convolved[:, :, 1], convolved[:, :, 2:]

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if prefill:
            # padding tokens have nonzero q/k/v/g via biases; zero g/beta so they don't pollute the recurrent-state sum and its decay.
            g = g * valid_mask
            beta = beta * valid_mask
        g = g.reshape(batch_size, seq_len, heads, ratio).permute(0, 2, 3, 1).contiguous()
        beta = beta.reshape(batch_size, seq_len, heads, ratio).permute(0, 2, 3, 1).contiguous()
        slot = torch.tensor(0, dtype=torch.int16) if slot is None else slot
        operands = (query.contiguous(), key.contiguous(), value.contiguous(), g, beta, recurrent_state, slot)
        if "prefill" in self._phase:
            carry = torch.ones(batch_size, dtype=value.dtype) if carry is None else carry
            core_attn_out = torch.ops.rbln_custom_ops.gdn_prefill(*operands, carry)
        else:
            core_attn_out = torch.ops.rbln_custom_ops.gdn_decode(*operands)

        core_attn_out = self.norm(core_attn_out, z).permute(0, 3, 1, 2, 4).reshape(batch_size, seq_len, -1)
        output = self.out_proj(core_attn_out)
        return output, new_conv_state.contiguous()


class Qwen3_5LinearDecoderLayer(nn.Module):
    """A ``linear_attention`` decoder layer: GatedDeltaNet token mixer + MLP (on-device static states)."""

    def __init__(self, layer: nn.Module, linear_attn: Qwen3_5GatedDeltaNet):
        super().__init__()
        self.linear_attn = linear_attn
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self._phase = "prefill"

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase: str):
        self._phase = phase
        self.linear_attn.phase = phase

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        query_position: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        slot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_conv_state = self.linear_attn(
            hidden_states,
            conv_state,
            recurrent_state,
            query_position=query_position,
            valid_mask=valid_mask,
            state_mask=state_mask,
            slot=slot,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, new_conv_state


class Qwen3_5Attention(DecoderOnlyAttention):
    def __post_init__(self, self_attn):
        self.k_proj = self_attn.k_proj
        self.v_proj = self_attn.v_proj
        self.o_proj = self_attn.o_proj
        self.q_norm = self_attn.q_norm
        self.k_norm = self_attn.k_norm

        hidden = self_attn.q_proj.weight.shape[1]
        has_bias = self_attn.q_proj.bias is not None
        w = self_attn.q_proj.weight.data.view(self.num_heads, 2, self.head_dim, hidden)
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=has_bias)
        self.gate_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=has_bias)
        self.q_proj.weight = nn.Parameter(w[:, 0].reshape(self.num_heads * self.head_dim, hidden).contiguous())
        self.gate_proj.weight = nn.Parameter(w[:, 1].reshape(self.num_heads * self.head_dim, hidden).contiguous())
        if has_bias:
            bsplit = self_attn.q_proj.bias.data.view(self.num_heads, 2, self.head_dim)
            self.q_proj.bias = nn.Parameter(bsplit[:, 0].reshape(-1).contiguous())
            self.gate_proj.bias = nn.Parameter(bsplit[:, 1].reshape(-1).contiguous())

        partial_rotary_factor = getattr(self.config, "partial_rotary_factor", 1.0)
        self.rotary_ndims = int(self.head_dim * partial_rotary_factor)

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        return apply_rotary_pos_emb_partial(query_states, key_states, cos, sin, ndim=self.rotary_ndims)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_positions: torch.LongTensor,
        past_key_values: tuple[tuple[torch.Tensor]],
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        block_tables: torch.Tensor | None = None,
        lora_int_id: torch.Tensor | None = None,
    ):
        batch_size, query_length, _ = hidden_states.size()

        gate = self.gate_proj(hidden_states)
        query_states = (
            self.q_proj(hidden_states).view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(batch_size, query_length, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(batch_size, query_length, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if cos is not None and sin is not None:
            query_states, key_states = self.apply_rotary_pos_embed(query_states, key_states, cos, sin)

        if batch_size > 1 and "prefill" in self.phase:
            raise NotImplementedError(f"batch size should be 1 if prefill phase, but got {batch_size}.")

        k_scale, v_scale = self.maybe_get_kvcache_scale()
        attn_output = self.get_attention_op()(
            query_states,
            key_states,
            value_states,
            attention_mask,
            past_key_state=past_key_values[self.layer_idx][0],
            past_value_state=past_key_values[self.layer_idx][1],
            seq_position=seq_positions,
            scale=self.scale,
            block_tables=block_tables,
            block_size=self.kvcache_block_size,
            k_scale=k_scale,
            v_scale=v_scale,
            s_aux=getattr(self, "sinks", None),
        )
        attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3_5Model(DecoderOnlyModel):
    """Hybrid decoder body: dispatches ``linear_attention`` vs ``full_attention`` per layer and
    threads the linear-attention state updates out as extra returns."""

    def __init__(self, model, layers, rbln_config, use_learned_pos_emb=None, use_rotary_emb=True):
        super().__init__(model, layers, rbln_config, use_learned_pos_emb, use_rotary_emb)
        self.linear_attention_layers = rbln_config.linear_attention_layers

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        query_position: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        past_states: tuple[tuple[torch.Tensor]] | None = None,
        rotary_emb: nn.Module | None = None,
        global_block_tables: torch.Tensor | None = None,
        local_block_tables: torch.Tensor | None = None,
        lora_int_id: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,  # prefill only: which max-batch cache slot this item uses
        output_hidden_states: bool | None = None,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.get_embedding()(input_ids)
        hidden_states = inputs_embeds * self.hidden_multiplier

        position_ids = position_ids if position_ids is not None else cache_position
        cos = sin = None
        if rotary_emb is not None:
            if isinstance(rotary_emb, torch.Tensor):
                cos, sin = rotary_emb[0], rotary_emb[1]
            else:
                cos, sin = rotary_emb(hidden_states, self.max_seq_len)
                cos, sin = slice_and_unsqueeze_cos_sin(cos, sin, position_ids)

        if self.attn_impl == "flash_attn":
            seq_positions = self.convert_sequence_positions_for_flash_attn(
                seq_positions=cache_position[:, 0], max_seq_len=self.max_seq_len
            )
        else:
            seq_positions = cache_position.amin(dim=1, keepdim=True)

        all_hidden_states = () if output_hidden_states else None
        new_states: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if layer_idx in self.linear_attention_layers:
                conv_state, recurrent_state = past_states[layer_idx]
                if batch_idx is not None:
                    # PREFILL processes ONE item -> take its slot [batch_idx] from the max-batch cache -> (1, ...).
                    # The recurrent cache goes in whole: `gdn_prefill` addresses its own rows by `slot`.
                    conv_in = conv_state[batch_idx.to(torch.int).unsqueeze(0)]
                    _pos = batch_idx.to(torch.int16)
                else:
                    conv_in = conv_state
                    _pos = torch.tensor(0, dtype=torch.int16)
                hidden_states, new_conv_state = layer(
                    hidden_states,
                    conv_in,
                    recurrent_state,
                    query_position=query_position,
                    valid_mask=valid_mask,
                    state_mask=state_mask,
                    slot=_pos,
                )
                # Only the conv cache is written back here; `gdn_prefill` / `gdn_decode` already wrote their
                # rows of the recurrent cache.
                _axis0 = torch.tensor(0, dtype=torch.int16)
                new_states.append(
                    torch.ops.rbln_custom_ops.rbln_cache_update(conv_state, new_conv_state, _pos, _axis0)
                )
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    seq_positions=seq_positions,
                    past_key_values=past_key_values,
                    cos=cos,
                    sin=sin,
                    block_tables=global_block_tables,
                    lora_int_id=lora_int_id,
                )

        hidden_states = self.get_last_layernorm()(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return hidden_states, new_states, all_hidden_states


class Qwen3_5ForCausalLM(DecoderOnlyForCausalLM):
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        query_position: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        past_states: tuple[tuple[torch.Tensor]] | None = None,
        rotary_emb: nn.Module | None = None,
        global_block_tables: torch.Tensor | None = None,
        local_block_tables: torch.Tensor | None = None,
        lora_int_id: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
    ):
        hidden_states, new_states, all_hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            past_states=past_states,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
            lora_int_id=lora_int_id,
            state_mask=state_mask,
            valid_mask=valid_mask,
            batch_idx=batch_idx,
            output_hidden_states=output_hidden_states,
        )

        if "prefill" in self.phase and query_position is not None:
            hidden_states = hidden_states[:, query_position.to(torch.int).unsqueeze(0)]

        logits = self.lm_head(hidden_states)
        return logits, new_states, all_hidden_states


class Qwen3_5_CausalLMWrapper(DecoderOnlyWrapper):
    _use_rotary_emb = True

    def get_rotary_emb(self, max_seq_len):
        config = copy.deepcopy(self.config)
        rope_params = dict(getattr(config, "rope_parameters", None) or {})
        if getattr(config, "rope_theta", None) is None:
            config.rope_theta = rope_params.get("rope_theta", 10000.0)
        if getattr(config, "partial_rotary_factor", None) is None:
            config.partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
        config.rope_scaling = None
        return RotaryEmbedding(config=config, max_seq_len_cached=max_seq_len)

    def get_rbln_attn_class(self):
        return Qwen3_5Attention

    def get_rbln_model_class(self):
        return Qwen3_5Model

    def get_rbln_causal_lm_class(self):
        return Qwen3_5ForCausalLM

    def convert_to_rbln_class(self, model, max_seq_len: int, use_rotary_emb: bool):
        layer_types = self.config.layer_types
        new_layers = []
        for layer_idx, layer in enumerate(self.get_decoder_layers(model)):
            if layer_types[layer_idx] == "linear_attention":
                rbln_deltanet = Qwen3_5GatedDeltaNet(layer.linear_attn, self.rbln_config, layer_idx)
                new_layers.append(Qwen3_5LinearDecoderLayer(layer, rbln_deltanet))
            else:
                new_self_attn = self.get_rbln_attn_class()(layer.self_attn, self.rbln_config, is_sliding=False)
                new_layers.append(
                    self.get_rbln_layer_class()(layer, new_self_attn, lora_config=self.rbln_config.lora_config)
                )

        new_model = self.get_rbln_model_class()(
            self.get_model_layer(model),
            new_layers,
            self.rbln_config,
            use_learned_pos_emb=self.__class__._use_learned_pos_emb,
            use_rotary_emb=use_rotary_emb,
        )
        if self.is_causal_lm:
            return self.get_rbln_causal_lm_class()(model, new_model)
        return new_model

    def _split_layer_states(self, pairs):
        # Split a per-layer list of pairs by layer type: full_attention -> past_key_values (key, value),
        # linear_attention -> past_states (conv_state, recurrent_state).
        linear = {i for i, t in enumerate(self.config.layer_types) if t == "linear_attention"}
        past_key_values = [pair if i not in linear else None for i, pair in enumerate(pairs)]
        past_states = [pair if i in linear else None for i, pair in enumerate(pairs)]
        return past_key_values, past_states

    def prepare_forward_args(self, *args):
        args = list(args)
        has_linear = any(t == "linear_attention" for t in self.config.layer_types)
        batch_idx = args.pop() if (has_linear and "prefill" in self.phase) else None
        valid_mask = args.pop() if has_linear else None
        state_mask = args.pop() if has_linear else None

        (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            query_position,
            attention_mask,
            position_ids,
            lora_int_id,
            pairs,  # base groups the trailing flat per-layer states into (a, b) pairs
            rotary_emb,
        ) = super().prepare_forward_args(*args)

        # full_attention -> past_key_values (key, value); linear_attention -> past_states (conv, recurrent).
        past_key_values, past_states = self._split_layer_states(pairs)
        return (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            query_position,
            attention_mask,
            position_ids,
            lora_int_id,
            past_key_values,
            past_states,
            rotary_emb,
            state_mask,
            valid_mask,
            batch_idx,
        )

    def forward(self, *args):
        (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            query_position,
            attention_mask,
            position_ids,
            lora_int_id,
            past_key_values,
            past_states,
            rotary_emb,
            state_mask,
            valid_mask,
            batch_idx,
        ) = self.prepare_forward_args(*args)

        logits, new_states, all_hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            past_states=past_states,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
            lora_int_id=lora_int_id,
            state_mask=state_mask,
            valid_mask=valid_mask,
            batch_idx=batch_idx,
            output_hidden_states=self.rbln_config.output_hidden_states,
        )

        # Linear-attention state updates are returned so the runtime can persist them on the host; the
        # per-layer hidden states (n_layers + 1) trail last when output_hidden_states is requested.
        if self.rbln_config.output_hidden_states:
            return (logits, *new_states, *all_hidden_states)
        return (logits, *new_states)


class Qwen3_5_LanguageModelWrapper(Qwen3_5_CausalLMWrapper):
    """The hybrid Qwen3.5 text backbone wired for the vision-language runtime."""

    _use_rotary_emb = False

    def __init__(self, model: "PreTrainedModel", rbln_config, use_rotary_emb: bool):
        original_config = model.config
        model.config = model.config.text_config
        super().__init__(model, rbln_config, use_rotary_emb)
        model.config = original_config

    def get_decoder_layers(self, model: "PreTrainedModel"):
        return model.get_decoder().layers

    def get_model_layer(self, model: "PreTrainedModel"):
        return model.get_decoder()

    def prepare_forward_args(self, *args):
        args = list(args)
        has_linear = any(t == "linear_attention" for t in self.config.layer_types)
        batch_idx = args.pop() if (has_linear and "prefill" in self.phase) else None
        valid_mask = args.pop() if has_linear else None
        state_mask = args.pop() if has_linear else None
        input_ids = None if self.rbln_config.use_inputs_embeds else args.pop(0)
        inputs_embeds = args.pop(0) if self.rbln_config.use_inputs_embeds else None
        cache_position = args.pop(0)
        global_block_tables = args.pop(0)
        local_block_tables = None
        position_embeds = args.pop(0)
        query_position = args.pop(0) if self.phase == "prefill" and self.rbln_config.logits_to_keep > 0 else None
        position_ids = None
        attention_mask = args.pop(0) if self.rbln_config.use_attention_mask else None
        lora_int_id = args.pop(0) if self.rbln_config.lora_config else None

        # full_attention -> past_key_values (key, value); linear_attention -> past_states (conv, recurrent).
        state_args = args
        if len(state_args) != 2 * self.num_hidden_layers:
            raise ValueError(f"Different states to model's config. {len(state_args)} != {2 * self.num_hidden_layers}")
        pairs = [[state_args[i * 2], state_args[i * 2 + 1]] for i in range(self.num_hidden_layers)]
        past_key_values, past_states = self._split_layer_states(pairs)

        return (
            input_ids,
            inputs_embeds,
            cache_position,
            global_block_tables,
            local_block_tables,
            query_position,
            attention_mask,
            position_ids,
            lora_int_id,
            past_key_values,
            past_states,
            position_embeds,  # cos/sin
            state_mask,
            valid_mask,
            batch_idx,
        )
