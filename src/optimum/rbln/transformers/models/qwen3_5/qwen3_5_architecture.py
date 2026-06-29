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

"""RBLN graph rewrite for Qwen3.5 (text backbone).

Qwen3.5 is a hybrid decoder: ``full_attention`` layers (gated softmax attention with a
paged KV cache, like Qwen3 + an output gate + partial RoPE) interleaved with
``linear_attention`` layers (GatedDeltaNet). Path A bring-up choices (see the project memo
`rbln-qwen35-deltanet-compile`):

- The GatedDeltaNet uses the *recurrent* delta rule for BOTH prefill and decode (the chunked
  kernel does not lower; the recurrent form compiles and is numerically validated on NPU).
- Its two states (``conv_state`` and ``recurrent_state``) are carried as ordinary graph
  inputs/outputs (functional), and a Qwen3.5-specific runtime keeps them on the host between
  prefill chunks / decode steps. Each linear layer reuses its two ``past_key_values`` slots to
  carry ``(conv_state, recurrent_state)`` instead of ``(key, value)``.
"""

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel

from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
    RotaryEmbedding,
    apply_rotary_pos_emb_partial,
    slice_and_unsqueeze_cos_sin,
)

from ..qwen3_vl.qwen3_vl_architecture import Qwen3VLVisionBlock
from .qwen3_5_architecture import Qwen3_5Wrapper

class Qwen3_5VisionModelWrapper(nn.Module):
    """Qwen3.5 vision encoder for RBLN: transformer blocks + merger, NO deepstack.

    Patch embedding, position-embed interpolation and rotary computation run on the host; the
    compiled graph takes the patch ``hidden_states`` plus the precomputed attention mask and
    rotary ``cos``/``sin``, and returns the merged image embeddings.
    """

    def __init__(self, model: nn.Module, rbln_config):
        super().__init__()
        self.merger = model.merger
        self.rbln_config = rbln_config
        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(block, rbln_config) for block in model.blocks])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = (1.0 - attn_mask) * torch.finfo(hidden_states.dtype).min
        for block in self.blocks:
            hidden_states = block(hidden_states, attn_mask, [cos, sin])
        return self.merger(hidden_states)


class Qwen3_5GatedDeltaNet(nn.Module):
    """Recurrent GatedDeltaNet token mixer for RBLN (functional: states in -> states out).

    conv_state is stored as ``(B, K-1, conv_dim)`` (innermost = conv_dim, a multiple of 64 as
    RBLN requires) and transposed to ``(B, conv_dim, K-1)`` only inside the math.
    """

    def __init__(self, linear_attn: nn.Module, rbln_config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self._phase = "prefill"

        # Reuse the original (trained) submodules / parameters.
        self.in_proj_qkv = linear_attn.in_proj_qkv
        self.in_proj_z = linear_attn.in_proj_z
        self.in_proj_b = linear_attn.in_proj_b
        self.in_proj_a = linear_attn.in_proj_a
        self.conv1d = linear_attn.conv1d
        self.norm = linear_attn.norm  # gated RMSNorm: norm(x) * silu(gate)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        k_1 = self.conv_kernel_size - 1

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (B, conv_dim, S)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Causal depthwise conv with the cached left-context prepended (unifies prefill & decode).
        conv_ctx = conv_state.transpose(1, 2)  # (B, conv_dim, K-1)
        x = torch.cat([conv_ctx, mixed_qkv], dim=-1)  # (B, conv_dim, (K-1)+S)
        new_conv_state = x[..., -k_1:].transpose(1, 2).contiguous()  # (B, K-1, conv_dim)
        conv_out = F.silu(F.conv1d(x, self.conv1d.weight, self.conv1d.bias, padding=0, groups=self.conv_dim))

        mixed_qkv = conv_out.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, new_recurrent_state = torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output = self.out_proj(core_attn_out)
        return output, new_conv_state, new_recurrent_state.contiguous()


class Qwen3_5LinearDecoderLayer(nn.Module):
    """A ``linear_attention`` decoder layer: GatedDeltaNet token mixer + MLP (functional state)."""

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_conv_state, new_recurrent_state = self.linear_attn(
            hidden_states, conv_state, recurrent_state
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, new_conv_state, new_recurrent_state


class Qwen3_5Attention(DecoderOnlyAttention):
    """Full-attention layer: Qwen3-style q/k-norm + an output gate + partial RoPE.

    ``q_proj`` emits ``num_heads * head_dim * 2`` and is split into (query, gate); the attention
    output is multiplied by ``sigmoid(gate)`` before ``o_proj``.
    """

    def __post_init__(self, self_attn):
        self.k_proj = self_attn.k_proj
        self.v_proj = self_attn.v_proj
        self.o_proj = self_attn.o_proj
        self.q_norm = self_attn.q_norm
        self.k_norm = self_attn.k_norm

        # qwen3.5 fuses (query, output-gate) into q_proj -> num_heads*head_dim*2, recovered by
        # `q_proj(h).view(.., num_heads, 2*head_dim).chunk(2, -1)`. We pre-split the weight into two
        # separate Linears so the traced graph has NO chunk/split op: a split whose BOTH halves are
        # consumed fails to lower on RBLN ("view op split fail"). Two matmuls lower cleanly.
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

        # Concrete Python int (NOT cos.shape[-1], which traces dynamically and breaks rotary lowering).
        partial_rotary_factor = getattr(self.config, "partial_rotary_factor", 1.0)
        self.rotary_ndims = int(self.head_dim * partial_rotary_factor)

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        return apply_rotary_pos_emb_partial(query_states, key_states, cos, sin, ndim=self.rotary_ndims)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_positions: torch.LongTensor,
        past_key_values: Tuple[Tuple[torch.Tensor]],
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        lora_int_id: Optional[torch.Tensor] = None,
    ):
        batch_size, query_length, _ = hidden_states.size()

        gate = self.gate_proj(hidden_states)  # (B, S, num_heads * head_dim)
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

        # Apply the output gate in the per-head 4D space (B, S, num_heads, head_dim) with concrete
        # dims; multiplying the custom-op output by the gate in flat 2048-d space makes RTOSA shape
        # inference fail ("inferred shape must be > 0").
        attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)
        # WORKAROUND ATTEMPT: route attn_output through a numerically-identity elementwise op
        # (2x - x == x) to give it a "standard op" output provenance before the gate multiply.
        # The direct `custom_op_output * gate` fails RTOSA shape inference.
        attn_output = 2.0 * attn_output - attn_output
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3_5Model(DecoderOnlyModel):
    """Hybrid decoder body: dispatches ``linear_attention`` vs ``full_attention`` per layer and
    threads the linear-attention state updates out as extra returns."""

    def __init__(self, model, layers, rbln_config, use_learned_pos_emb=None, use_rotary_emb=True):
        super().__init__(model, layers, rbln_config, use_learned_pos_emb, use_rotary_emb)
        self.linear_attention_layers = set(rbln_config.linear_attention_layers)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        query_position: torch.Tensor = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        rotary_emb: Optional[nn.Module] = None,
        global_block_tables: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
        lora_int_id: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
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
                # VL path: mRoPE cos/sin are precomputed on the host and passed in as a
                # stacked tensor (rotary_emb[0]=cos, rotary_emb[1]=sin); no inline rotary.
                cos, sin = rotary_emb[0], rotary_emb[1]
            else:
                cos, sin = rotary_emb(hidden_states, self.max_seq_len)
                cos, sin = slice_and_unsqueeze_cos_sin(cos, sin, position_ids)

        if self.attn_impl == "flash_attn":
            seq_positions = self.convert_sequence_positions_for_flash_attn(
                seq_positions=cache_position[:, 0], max_seq_len=self.max_seq_len
            )
        else:
            seq_positions = cache_position[:, :1]

        all_hidden_states = () if output_hidden_states else None
        new_states: List[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if layer_idx in self.linear_attention_layers:
                conv_state, recurrent_state = past_key_values[layer_idx]
                hidden_states, new_conv_state, new_recurrent_state = layer(hidden_states, conv_state, recurrent_state)
                new_states.append(new_conv_state)
                new_states.append(new_recurrent_state)
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
        return hidden_states, all_hidden_states, new_states


class Qwen3_5ForCausalLM(DecoderOnlyForCausalLM):
    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        query_position: torch.Tensor = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        rotary_emb: nn.Module = None,
        global_block_tables: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
        lora_int_id: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        hidden_states, all_hidden_states, new_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
            lora_int_id=lora_int_id,
            output_hidden_states=output_hidden_states,
        )

        if "prefill" in self.phase and query_position is not None:
            hidden_states = hidden_states[:, query_position.to(torch.int).unsqueeze(0)]

        logits = self.lm_head(hidden_states)
        return logits, all_hidden_states, new_states


class Qwen3_5_CausalLMWrapper(DecoderOnlyWrapper):
    # for Causal LM
    _use_rotary_emb = True

    def get_rotary_emb(self, max_seq_len):
        # Text-only path: Qwen3.5's mRoPE reduces to standard partial RoPE (rotary dim =
        # head_dim * partial_rotary_factor). Normalize the rope attrs the base RotaryEmbedding reads.
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
            rotary_emb,
        ) = self.prepare_forward_args(*args)

        logits, all_hidden_states, new_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
            lora_int_id=lora_int_id,
            output_hidden_states=self.rbln_config.output_hidden_states,
        )

        # Linear-attention state updates are returned so the runtime can persist them on the host.
        if self.rbln_config.output_hidden_states:
            return (logits, *new_states, *all_hidden_states)
        return (logits, *new_states)


class Qwen3_5_LanguageModelWrapper(Qwen3_5_CausalLMWrapper):
    """The hybrid Qwen3.5 text backbone wired for the VL runtime.

    Reuses ``Qwen3_5Wrapper``'s hybrid graph rewrite (``convert_to_rbln_class``, the
    ``get_rbln_*`` factories that emit GatedDeltaNet linear layers + gated full-attention
    layers, and the linear-state threading in ``Qwen3_5Model``). The only changes vs the
    text-only wrapper:

    - ``model.config`` is a ``Qwen3_5Config`` (vision + text); swap it to ``text_config`` for
      the parent ``DecoderOnlyWrapper`` initialization (which expects text attributes).
    - the language model is reached via ``model.get_decoder()`` (it is nested under the VL model).
    - ``position_embeds`` (precomputed mRoPE cos/sin) is an explicit graph input, passed to the
      model as the ``rotary_emb`` tensor; there is no inline ``RotaryEmbedding`` and no deepstack.
    """

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

        # 2 state slots per layer: (conv_state, recurrent_state) for linear_attention layers,
        # (key, value) for full_attention layers. The model dispatches per layer_idx.
        past_key_values = args
        if len(past_key_values) != 2 * self.num_hidden_layers:
            raise ValueError(
                f"Different past_key_values to model's config. {len(past_key_values)} != {2 * self.num_hidden_layers}"
            )
        _past_key_values = []
        for i in range(self.num_hidden_layers):
            _past_key_values.append([past_key_values[i * 2], past_key_values[i * 2 + 1]])
        past_key_values = _past_key_values

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
            position_embeds,
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
            position_embeds,
        ) = self.prepare_forward_args(*args)

        logits, all_hidden_states, new_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            query_position=query_position,
            past_key_values=past_key_values,
            rotary_emb=position_embeds,  # precomputed mRoPE (cos, sin); see Qwen3_5Model.forward
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
            lora_int_id=lora_int_id,
            output_hidden_states=self.rbln_config.output_hidden_states,
        )

        # Linear-attention state updates are returned so the runtime can persist them.
        if self.rbln_config.output_hidden_states:
            return (logits, *new_states, *all_hidden_states)
        return (logits, *new_states)