# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..decoderonly.configuration_decoderonly import RBLNLoRAConfig
from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyLayer,
    DecoderOnlyWrapper,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_partial,
)


class Qwen3NextWrapper(DecoderOnlyWrapper):
    def get_rbln_layer_class(self):
        return Qwen3NextLayer

    def convert_to_rbln_class(self, model, max_seq_len: int):
        new_layers = []
        layers = self.get_decoder_layers(model)

        for layer_idx, layer in enumerate(layers):
            layer_type = getattr(layer, "layer_type", None) or getattr(self.config, "layer_types", [None])[layer_idx]

            if layer_type == "linear_attention" and hasattr(layer, "linear_attn"):
                # linear attention
                new_token_mixer = Qwen3NextLinearAttention(layer.linear_attn, self.rbln_config, layer_idx)
            else:
                # full attention
                new_token_mixer = Qwen3NextGatedAttention(layer.self_attn, self.rbln_config, layer_idx)

            new_layer = self.get_rbln_layer_class()(layer, new_token_mixer, lora_config=self.rbln_config.lora_config)
            new_layers.append(new_layer)

        new_model = super().get_rbln_model_class()(
            self.get_model_layer(model),
            new_layers,
            self.rbln_config,
            use_learned_pos_emb=self.__class__._use_learned_pos_emb,
        )
        if self.is_causal_lm:
            new_model = super().get_rbln_causal_lm_class()(model, new_model)
        return new_model

    def prepare_forward_args(self, *args):
        """
        Extends the base decoderonly argument parsing by accepting additional
        per-linear-layer recurrent state caches after the standard past_key_values.
        The additional cache tensors are attached as a 3rd entry on each layer's past_key_values list:
            past_key_values[layer_idx] == [k_cache, v_cache, state_cache] for linear layers
        """
        args = list(args)

        input_ids = None if self.rbln_config.use_inputs_embeds else args.pop(0)
        inputs_embeds = args.pop(0) if self.rbln_config.use_inputs_embeds else None
        cache_position = args.pop(0)
        global_block_tables = args.pop(0) if self.rbln_config.use_global_attention else None
        local_block_tables = args.pop(0) if self.rbln_config.use_local_attention else None

        if "prefill" in self.phase and (self.rbln_config.logits_to_keep == 1 or self.rbln_config.use_local_attention):
            _ = args.pop(0)
            query_position = None
        else:
            query_position = None
        attention_mask = args.pop(0) if self.rbln_config.use_attention_mask else None
        position_ids = args.pop(0) if self.rbln_config.use_position_ids else None
        lora_int_id = args.pop(0) if self.rbln_config.lora_config else None

        num_layers = self.config.num_hidden_layers

        layer_types = getattr(self.config, "layer_types", None) or []
        linear_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
        kv_layers = [i for i, t in enumerate(layer_types) if t != "linear_attention"]

        expected_kv = 2 * len(kv_layers)
        kv_flat = args[:expected_kv]
        extra = args[expected_kv:]

        # KV layers: [k_cache, v_cache]
        # linear layers: [state_cache]
        past_key_values: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]

        kv_it = iter(kv_flat)
        for layer_idx in kv_layers:
            past_key_values[layer_idx] = [next(kv_it), next(kv_it)]

        for state_tensor, layer_idx in zip(extra, linear_layers, strict=False):
            past_key_values[layer_idx] = [state_tensor]

        if hasattr(self, "rotary_emb_global") and hasattr(self, "rotary_emb_local"):
            rotary_emb = (self.rotary_emb_global, self.rotary_emb_local)
        else:
            rotary_emb = self.rotary_emb

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
            rotary_emb,
        )


class Qwen3NextLayer(DecoderOnlyLayer):
    def __init__(self, layer, self_attn: DecoderOnlyAttention, lora_config: Optional[RBLNLoRAConfig] = None):
        super().__init__(layer, self_attn, lora_config)
        self.mlp = (
            Qwen3NextSparseMoeBlock(self._original_mod.mlp)
            if self._original_mod.mlp.__class__.__name__ == "Qwen3NextSparseMoeBlock"
            else self._original_mod.mlp
        )

    def get_mlp(self) -> nn.Module:
        return self.mlp


class Qwen3NextGatedAttention(nn.Module):
    def __init__(self, self_attn, rbln_config, layer_idx: int):
        super().__init__()
        from ..decoderonly.decoderonly_architecture import AttentionOp, FlashAttentionOp

        self._original_mod = self_attn
        self.rbln_config = rbln_config
        self.layer_idx = layer_idx
        self._phase = "prefill"

        self.config = self_attn.config
        self.num_heads = self_attn.config.num_attention_heads
        self.num_key_value_heads = self_attn.config.num_key_value_heads
        self.head_dim = self_attn.head_dim
        self.q_proj = self_attn.q_proj
        self.k_proj = self_attn.k_proj
        self.v_proj = self_attn.v_proj
        self.o_proj = self_attn.o_proj
        self.q_norm = self_attn.q_norm
        self.k_norm = self_attn.k_norm

        self.scale = torch.nn.Parameter(torch.tensor(1 / math.sqrt(self.head_dim)))

        self.attn_impl = rbln_config.attn_impl
        self.kvcache_partition_len = getattr(rbln_config, "kvcache_partition_len", None)
        self.kvcache_block_size = rbln_config.kvcache_block_size

        if self.attn_impl == "flash_attn":
            self.attn_op = FlashAttentionOp(
                self.num_heads,
                self.head_dim,
                self.num_key_value_heads,
                self.kvcache_partition_len,
                rbln_config=rbln_config,
                is_sliding=False,
            )
        else:
            self.attn_op = AttentionOp(
                self.num_heads,
                self.head_dim,
                self.num_key_value_heads,
                rbln_config=rbln_config,
                is_sliding=False,
            )

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase: str):
        self._phase = phase
        self.attn_op.phase = phase

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_positions: torch.LongTensor,
        past_key_values,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        lora_int_id: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(batch_size, seq_len, -1)  # [B,S,H*D]

        query_states = self.q_norm(query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)).transpose(
            1, 2
        )
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)
        value_states = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        if cos is not None and sin is not None:
            rotary_dim = cos.shape[-1]
            if rotary_dim != query_states.shape[-1]:
                query_states, key_states = apply_rotary_pos_emb_partial(
                    query_states, key_states, cos, sin, ndim=rotary_dim
                )
            else:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_out = self.attn_op(
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
            k_scale=getattr(getattr(self.k_proj, "k_scale", None), "data", None),
            v_scale=getattr(getattr(self.v_proj, "v_scale", None), "data", None),
            s_aux=getattr(self, "sinks", None),
        )

        attn_out = attn_out * torch.sigmoid(gate)
        return self.o_proj(attn_out)


class Qwen3NextLinearAttention(nn.Module):
    def __init__(self, linear_attn_mod, rbln_config, layer_idx: int):
        super().__init__()
        self._original_mod = linear_attn_mod
        self.rbln_config = rbln_config
        self.layer_idx = layer_idx
        self._phase = "prefill"

        self.hidden_size = linear_attn_mod.hidden_size
        self.num_v_heads = linear_attn_mod.num_v_heads
        self.num_k_heads = linear_attn_mod.num_k_heads
        self.head_k_dim = linear_attn_mod.head_k_dim
        self.head_v_dim = linear_attn_mod.head_v_dim
        self.key_dim = linear_attn_mod.key_dim
        self.value_dim = linear_attn_mod.value_dim
        self.conv_kernel_size = linear_attn_mod.conv_kernel_size

        self.in_proj_qkvz = linear_attn_mod.in_proj_qkvz
        self.in_proj_ba = linear_attn_mod.in_proj_ba
        self.dt_bias = linear_attn_mod.dt_bias
        self.A_log = linear_attn_mod.A_log
        self.conv1d = linear_attn_mod.conv1d
        self.out_proj = linear_attn_mod.out_proj
        self.norm = linear_attn_mod.norm

        self.scale = torch.nn.Parameter(torch.tensor(1 / math.sqrt(self.head_k_dim)))

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase: str):
        self._phase = phase

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_positions: torch.LongTensor,
        past_key_values,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        lora_int_id: Optional[torch.Tensor] = None,
    ):
        if attention_mask is not None and attention_mask.dim() == 2:
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)

        batch_size, seq_len, _ = hidden_states.shape
        rep = self.num_v_heads // self.num_k_heads
        query = torch.ones(
            (batch_size, seq_len, self.num_k_heads, self.head_k_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        key = torch.ones_like(query)
        value = torch.ones(
            (batch_size, seq_len, self.num_k_heads, rep * self.head_v_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        z = torch.ones_like(value)
        b = torch.ones(
            (batch_size, seq_len, self.num_k_heads, rep),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        a = torch.ones_like(b)

        # projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        # projected_states_ba = self.in_proj_ba(hidden_states)

        # new_tensor_shape_qkvz = projected_states_qkvz.size()[:-1] + (
        #     self.num_k_heads,
        #     2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
        # )
        # new_tensor_shape_ba = projected_states_ba.size()[:-1] + (
        #     self.num_k_heads,
        #     2 * self.num_v_heads // self.num_k_heads,
        # )
        # mixed_qkvz = projected_states_qkvz.view(*new_tensor_shape_qkvz)
        # mixed_ba = projected_states_ba.view(*new_tensor_shape_ba)

        # split_arg_list_qkvz = [
        #     self.head_k_dim,
        #     self.head_k_dim,
        #     (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        #     (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        # ]
        # split_arg_list_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]
        # query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        # b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)

        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), b.size(1), self.num_v_heads)

        query, key, value_flat = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))
        mixed_qkv = torch.cat((query, key, value_flat), dim=-1).transpose(1, 2)  # [B, C, S]

        if self.phase == "decode":
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value_flat = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value_flat.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        qn = torch.rsqrt((query.float() * query.float()).sum(dim=-1, keepdim=True) + 1e-6)
        kn = torch.rsqrt((key.float() * key.float()).sum(dim=-1, keepdim=True) + 1e-6)
        query = (query.float() * qn).to(hidden_states.dtype)
        key = (key.float() * kn).to(hidden_states.dtype)

        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        g = g.transpose(1, 2).contiguous()
        beta = beta.transpose(1, 2).contiguous()

        # q/k/v: [B,H,S,D] -> [B,H,S,D/64,64]
        if (self.head_k_dim % 64) != 0 or (self.head_v_dim % 64) != 0:
            raise ValueError("linear head dims must be multiples of 64 for blocked kernels.")
        dk_blk = self.head_k_dim // 64
        dv_blk = self.head_v_dim // 64
        query_blk = query.view(batch_size, self.num_v_heads, seq_len, dk_blk, 64).contiguous()
        key_blk = key.view(batch_size, self.num_v_heads, seq_len, dk_blk, 64).contiguous()
        value_blk = value.view(batch_size, self.num_v_heads, seq_len, dv_blk, 64).contiguous()

        # g/beta: [B,H,S] -> [B,H,S/64,64] (prefill) or [B,H,1,64] (decode)
        if seq_len == 1:
            g_blk = F.pad(g, (0, 63)).view(batch_size, self.num_v_heads, 1, 64).contiguous()
            beta_blk = F.pad(beta, (0, 63)).view(batch_size, self.num_v_heads, 1, 64).contiguous()
        else:
            if (seq_len % 64) != 0:
                raise ValueError(f"prefill seq_len must be a multiple of 64 (got {seq_len})")
            g_blk = g.view(batch_size, self.num_v_heads, seq_len // 64, 64).contiguous()
            beta_blk = beta.view(batch_size, self.num_v_heads, seq_len // 64, 64).contiguous()

        pkv = past_key_values[self.layer_idx]
        if len(pkv) == 1:
            state_cache = pkv[0]
        elif len(pkv) >= 3:
            state_cache = pkv[2]
        else:
            raise ValueError("Missing linear state_cache tensor for linear_attention layer.")

        if block_tables is None:
            raise ValueError("block_tables must be provided for paged_linear_attn_*")

        if self.phase == "decode":
            core_attn_out = torch.ops.rbln_custom_ops.paged_linear_attn_decode(
                query_blk, key_blk, value_blk, g_blk, beta_blk, state_cache, self.scale, block_tables
            )
        else:
            core_attn_out = torch.ops.rbln_custom_ops.paged_linear_attn_prefill(
                query_blk, key_blk, value_blk, g_blk, beta_blk, state_cache, self.scale, block_tables
            )

        # core_attn_out is blocked: [B,S,H,Dv/64,64] -> [B,S,H,Dv]
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        z_shape_og = z.shape  # [B,S,H,Dv]

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z2 = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z2)
        core_attn_out = core_attn_out.reshape(z_shape_og).reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out)


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.num_experts = model.num_experts
        self.top_k = model.top_k
        self.norm_topk_prob = model.norm_topk_prob
        self.gate = model.gate
        self.shared_expert = model.shared_expert
        self.shared_expert_gate = model.shared_expert_gate
        self.experts = Qwen3NextMLP(model.experts, self.top_k, self.norm_topk_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits)
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            torch.nn.functional.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )
        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class Qwen3NextMLP(nn.Module):
    def __init__(self, expert_list, top_k, norm_topk_prob):
        super().__init__()
        self.hidden_size = expert_list[0].hidden_size
        self.intermediate_size = expert_list[0].intermediate_size
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

        self.num_experts = len(expert_list)
        self.gate_proj = nn.Linear(self.hidden_size, self.num_experts * self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.num_experts * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.num_experts * self.intermediate_size, self.hidden_size, bias=False)
        self.gate_proj.weight.data = torch.stack([expert.gate_proj.weight.data for expert in expert_list], dim=0)
        self.up_proj.weight.data = torch.stack([expert.up_proj.weight.data for expert in expert_list], dim=0)
        self.down_proj.weight.data = torch.stack([expert.down_proj.weight.data for expert in expert_list], dim=0)

    def forward(self, x, router_logits):
        return torch.ops.rbln_custom_ops.custom_moe_glu(
            x,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
            router_logits,
            self.top_k,
            self.norm_topk_prob,
        )
