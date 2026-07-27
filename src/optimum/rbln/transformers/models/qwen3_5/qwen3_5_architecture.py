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
import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

from ..decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_partial,
    slice_and_unsqueeze_cos_sin,
)


class Qwen3_5VisionAttention(nn.Module):
    """Qwen3.5 vision attention.

    Full (non-windowed) SDPA over the padded patch window; rotary applied on the precomputed host cos/sin.
    """

    def __init__(self, model: nn.Module, rbln_config) -> None:
        super().__init__()
        self._origin_model = model
        self.rbln_config = rbln_config
        self.num_heads = model.num_heads
        self.head_dim = getattr(model, "head_dim", model.proj.in_features // model.num_heads)
        self.qkv = model.qkv
        self.proj = model.proj
        self.scale = torch.tensor(1 / math.sqrt(self.head_dim), dtype=rbln_config.dtype)

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
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_output = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale.item(),
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(1, seq_length, -1)
        attn_output = self.proj(attn_output).squeeze(0)
        return attn_output


class Qwen3_5VisionBlock(nn.Module):
    """Qwen3.5 vision transformer block: (norm1 -> attn) + (norm2 -> mlp) residuals."""

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
    """Qwen3.5 vision encoder for RBLN: transformer blocks + merger, NO deepstack.

    Patch embedding, position-embed interpolation and rotary computation run on the host; the
    compiled graph takes the patch ``hidden_states`` plus the precomputed attention mask and
    rotary ``cos``/``sin``, and returns the merged image embeddings.
    """

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
        cos, sin = cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)
        for block in self.blocks:
            hidden_states = block(hidden_states, attn_mask, (cos, sin))
        return self.merger(hidden_states)


def rbln_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    tril_incl,
    tril_strict,
    initial_state,
    prefill_chunk_size,
    chunk_size,
    use_qk_l2norm_in_kernel=False,
):
    """Gated delta rule for RBLN PREFILL (chunk-parallel; HF ``torch_chunk_gated_delta_rule`` rewritten to
    lower on RBLN). A prefill window is split into ``n_chunks = prefill_chunk_size // chunk_size`` sub-chunks:
    intra-chunk terms (``decay_mask`` / ``T = (I-A)^-1`` / ``value`` / ``k_cumdecay``) are batched over the
    chunk axis, and the ``for`` loop carries ``recurrent_state`` between sub-chunks (inter-chunk). Inputs
    query/key ``(B,S,Hv,Dk)``, value ``(B,S,Hv,Dv)``, g/beta ``(B,S,Hv)``, initial_state the 3D cache layout
    ``(B,Hv,Dk*Dv)`` (reshaped to 4D internally after the mask de-statics it); returns core ``(B,S,Hv,Dv)`` and
    final state ``(B,Hv,Dk*Dv)``.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(initial_dtype) for x in (query, key, value, beta, g)
    ]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    gcs = chunk_size
    n_chunks = prefill_chunk_size // chunk_size
    query, key, value, v_beta, k_beta = [
        torch.stack([x[:, :, c * gcs : (c + 1) * gcs] for c in range(n_chunks)], dim=2)
        for x in (query, key, value, v_beta, k_beta)
    ]
    incr = torch.stack([g[:, :, c * gcs : (c + 1) * gcs] for c in range(n_chunks)], dim=2)
    g = incr.cumsum(dim=-1)

    # intra-chunk, batched over the chunk axis
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)) * tril_incl).exp() * tril_incl
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * tril_strict
    attn = torch.ops.rbln.tri_recur_update(attn)
    eye = tril_incl - tril_strict
    attn = attn + eye

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    # initial_state arrives in the 3D cache layout (B, Hv*Dk, Dv); it was masked on GatedDeltaNet
    # entry, so reshaping it to 4D (B, Hv, Dk, Dv) here is safe.
    last_recurrent_state = initial_state.reshape(
        initial_state.shape[0], query.shape[1], query.shape[-1], value.shape[-1]
    )

    # inter-chunk: sequential carry across sub-chunks.
    core_chunks = []
    for i in range(0, n_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_intra = (q_i @ k_i.transpose(-1, -2)) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i].exp().unsqueeze(-1)) @ last_recurrent_state
        core_chunks.append(attn_inter + attn_intra @ v_new)

        incr_i = incr[:, :, i]
        g_total = incr_i.sum(dim=-1, keepdim=True)
        decay_to_end = incr_i @ tril_strict[0, 0, 0]
        last_recurrent_state = (
            last_recurrent_state * g_total.unsqueeze(-1).exp()
            + (k_i * decay_to_end.exp().unsqueeze(-1)).transpose(-1, -2) @ v_new
        )

    # concat sub-chunk outputs along seq
    core = torch.cat(core_chunks, dim=2)
    core = core.transpose(1, 2).contiguous()
    # return the final state in the 3D cache layout (B, Hv*Dk, Dv) to match the static cache.
    last_recurrent_state = last_recurrent_state.reshape(
        last_recurrent_state.shape[0],
        last_recurrent_state.shape[1] * last_recurrent_state.shape[2],
        last_recurrent_state.shape[3],
    )
    return core, last_recurrent_state


def rbln_recurrent_gated_delta_rule_step(query, key, value, g, beta, initial_state, use_qk_l2norm_in_kernel=False):
    """Single-step (decode, seq=1) gated delta rule for RBLN. Numerically identical to HF
    ``torch_recurrent_gated_delta_rule`` at S=1, but rewritten to avoid ops that lower to garbage on device
    (cos≈0.2): the per-position output index-assign (ScatterInfo), the ``query[:, :, i]`` dynamic StridedSlice,
    and the non-innermost ``.sum(dim=-2)`` reductions (rewritten as matmuls). The S=1 axis is dropped with a
    reshape. Inputs query/key ``(B,1,Hv,Dk)``, value ``(B,1,Hv,Dv)``, g/beta ``(B,1,Hv)``, initial_state the
    3D cache layout ``(B,Hv,Dk*Dv)`` -> core ``(B,1,Hv,Dv)``, new_state ``(B,Hv,Dk*Dv)``. The state is kept 3D
    in the cache and reshaped to 4D here only after a compute (``* g_t``).
    """
    initial_dtype = query.dtype
    batch_size, _, num_v_heads, k_head_dim = query.shape
    v_head_dim = value.shape[-1]
    q = query.reshape(batch_size, num_v_heads, k_head_dim)
    k = key.reshape(batch_size, num_v_heads, k_head_dim)
    v = value.reshape(batch_size, num_v_heads, v_head_dim)

    def _l2norm_dot(x):
        # same as HF l2norm, but ||x||² via matmul dot-product instead of `(x*x).sum(-1)`: on the tiny seq=1
        # decode tensors RBLN lowers the innermost-axis sum to ~0 -> rsqrt(eps) blows up; matmul lowers correctly.
        ss = torch.matmul(x.unsqueeze(-2), x.unsqueeze(-1)).squeeze(-1)
        return x * torch.rsqrt(ss + 1e-6)

    if use_qk_l2norm_in_kernel:
        q = _l2norm_dot(q)
        k = _l2norm_dot(k)
    q_row = (q * (k_head_dim**-0.5)).unsqueeze(-2)
    k_row = k.unsqueeze(-2)
    v_row = v.unsqueeze(-2)
    beta_t = beta.reshape(batch_size, num_v_heads, 1, 1)

    # initial_state is the 3D cache (B, Hv*Dk, Dv). Fold the g decay into a per-(Hv*Dk) column vector and
    # multiply (this de-statics the read) THEN reshape to 4D (B, Hv, Dk, Dv)
    decay = (
        g.reshape(batch_size, num_v_heads, 1)
        .exp()
        .repeat(1, 1, k_head_dim)
        .reshape(batch_size, num_v_heads * k_head_dim, 1)
    )
    state = (initial_state * decay).reshape(batch_size, num_v_heads, k_head_dim, v_head_dim)
    kv_mem = torch.matmul(k_row, state)
    delta = (v_row - kv_mem) * beta_t
    new_state = state + torch.matmul(k_row.transpose(-1, -2), delta)
    core = torch.matmul(q_row, new_state)
    core = core.reshape(batch_size, 1, num_v_heads, v_head_dim).to(initial_dtype)
    # back to the 3D cache layout (B, Hv*Dk, Dv)
    new_state = new_state.reshape(batch_size, num_v_heads * k_head_dim, v_head_dim).to(initial_dtype)
    return core, new_state


class Qwen3_5GatedDeltaNet(nn.Module):
    """GatedDeltaNet token mixer for RBLN (conv_state + recurrent_state are on-device static caches).

    PREFILL uses the parallel chunked delta rule (``rbln_chunk_gated_delta_rule``, which lowers on RBLN);
    DECODE (seq=1) uses the recurrent delta rule. Both consume/return the same state layout so a
    chunk-prefill seamlessly hands its ``recurrent_state`` to recurrent-decode.

    conv_state is stored as ``(B, K-1, conv_dim)`` (innermost = conv_dim, a multiple of 64 as
    RBLN requires) and transposed to ``(B, conv_dim, K-1)`` only inside the math.
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

        # Pre-split the fused in_proj_qkv into separate Q/K/V projections (weight partition, numerically
        # identical): a depthwise conv is per-channel, so the channel-axis split commutes with it.
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

        self.prefill_chunk_size = getattr(rbln_config, "prefill_chunk_size", 128)
        # GDN sub-chunk: each window splits into `chunk_size` sub-chunks (must divide prefill_chunk_size);
        self.chunk_size = getattr(rbln_config, "gdn_chunk_size", None) or self.prefill_chunk_size

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
        conv_state_mask: torch.Tensor | None = None,
        recurrent_state_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        k_1 = self.conv_kernel_size - 1
        prefill = "prefill" in self._phase and valid_mask is not None

        # PREFILL window 0 starts from zero carried state: the runtime feeds a zeros mask for window 0 and a
        # ones mask afterward (same shape as the states -> plain elementwise multiply). Decode always carries.
        if prefill:
            if conv_state_mask is not None:
                conv_state = conv_state * conv_state_mask
            if recurrent_state_mask is not None:
                recurrent_state = recurrent_state * recurrent_state_mask

        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        kd, vd = self.key_dim, self.value_dim
        q_in = self.in_proj_q(hidden_states)
        k_in = self.in_proj_k(hidden_states)
        v_in = self.in_proj_v(hidden_states)

        q_cf = torch.cat([conv_state[:, :, :kd].transpose(1, 2), q_in.transpose(1, 2)], dim=-1)
        k_cf = torch.cat([conv_state[:, :, kd : 2 * kd].transpose(1, 2), k_in.transpose(1, 2)], dim=-1)
        v_cf = torch.cat([conv_state[:, :, 2 * kd :].transpose(1, 2), v_in.transpose(1, 2)], dim=-1)
        x_cf = torch.cat([q_cf, k_cf, v_cf], dim=1)

        if prefill:
            # new conv_state = the last K-1 conv INPUTS. In PREFILL the window is right-padded, so the last K-1 cols
            # are nonzero padding (via projection biases); select the last K-1 VALID cols via query_position.
            states = [x_cf[:, :, query_position.to(torch.int).unsqueeze(0) + i] for i in range(1, k_1 + 1)]
            new_conv_state = torch.cat(states, dim=2).transpose(1, 2).contiguous()
        else:
            new_conv_state = x_cf[:, :, -k_1:].transpose(1, 2).contiguous()

        _cw, _cb = self.conv1d.weight, self.conv1d.bias
        query = F.silu(
            F.conv1d(q_cf, _cw[:kd], _cb[:kd] if _cb is not None else None, padding=0, groups=kd).transpose(1, 2)
        )
        key = F.silu(
            F.conv1d(
                k_cf, _cw[kd : 2 * kd], _cb[kd : 2 * kd] if _cb is not None else None, padding=0, groups=kd
            ).transpose(1, 2)
        )
        value = F.silu(
            F.conv1d(v_cf, _cw[2 * kd :], _cb[2 * kd :] if _cb is not None else None, padding=0, groups=vd).transpose(
                1, 2
            )
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias)
        if prefill:
            # padding tokens have nonzero q/k/v/g via biases; zero g/beta so they don't pollute the recurrent-state sum and its decay.
            g = g * valid_mask
            beta = beta * valid_mask
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if "prefill" in self._phase:
            # Triangular masks built
            _cshape = (1, 1, 1, self.chunk_size, self.chunk_size)
            chunk_tril_incl = torch.tril(torch.ones(_cshape, device=query.device, dtype=query.dtype), diagonal=0)
            chunk_tril_strict = torch.tril(torch.ones(_cshape, device=query.device, dtype=query.dtype), diagonal=-1)
            core_attn_out, new_recurrent_state = rbln_chunk_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                chunk_tril_incl,
                chunk_tril_strict,
                recurrent_state,
                self.prefill_chunk_size,
                self.chunk_size,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, new_recurrent_state = rbln_recurrent_gated_delta_rule_step(
                query, key, value, g, beta, recurrent_state, use_qk_l2norm_in_kernel=True
            )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output = self.out_proj(core_attn_out)
        return output, new_conv_state, new_recurrent_state.contiguous()


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
        conv_state_mask: torch.Tensor | None = None,
        recurrent_state_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_conv_state, new_recurrent_state = self.linear_attn(
            hidden_states,
            conv_state,
            recurrent_state,
            query_position=query_position,
            valid_mask=valid_mask,
            conv_state_mask=conv_state_mask,
            recurrent_state_mask=recurrent_state_mask,
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
        self.linear_attention_layers = {i for i, l in enumerate(layers) if isinstance(l, Qwen3_5LinearDecoderLayer)}

    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        query_position: torch.Tensor = None,
        past_key_values: tuple[tuple[torch.Tensor]] = None,
        past_states: tuple[tuple[torch.Tensor]] = None,
        rotary_emb: nn.Module | None = None,
        global_block_tables: torch.Tensor | None = None,
        local_block_tables: torch.Tensor | None = None,
        lora_int_id: torch.Tensor | None = None,
        conv_state_mask: torch.Tensor | None = None,
        recurrent_state_mask: torch.Tensor | None = None,
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
                slotted = batch_idx is not None
                if slotted:
                    # PREFILL processes ONE item -> take its slot [batch_idx] from the max-batch cache -> (1, ...).
                    conv_in = conv_state[batch_idx.to(torch.int).unsqueeze(0)]
                    recurrent_in = recurrent_state[batch_idx.to(torch.int).unsqueeze(0)]
                    _pos = batch_idx.to(torch.int16)
                else:
                    conv_in, recurrent_in = conv_state, recurrent_state
                    _pos = torch.tensor(0, dtype=torch.int16)
                hidden_states, new_conv_state, new_recurrent_state = layer(
                    hidden_states,
                    conv_in,
                    recurrent_in,
                    query_position=query_position,
                    valid_mask=valid_mask,
                    conv_state_mask=conv_state_mask,
                    recurrent_state_mask=recurrent_state_mask,
                )
                _axis0 = torch.tensor(0, dtype=torch.int16)
                new_states.append(
                    torch.ops.rbln_custom_ops.rbln_cache_update(conv_state, new_conv_state, _pos, _axis0)
                )
                new_states.append(
                    torch.ops.rbln_custom_ops.rbln_cache_update(recurrent_state, new_recurrent_state, _pos, _axis0)
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
        past_key_values: tuple[tuple[torch.Tensor]] = None,
        past_states: tuple[tuple[torch.Tensor]] = None,
        rotary_emb: nn.Module = None,
        global_block_tables: torch.Tensor | None = None,
        local_block_tables: torch.Tensor | None = None,
        lora_int_id: torch.Tensor | None = None,
        conv_state_mask: torch.Tensor | None = None,
        recurrent_state_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
    ):
        hidden_states, all_hidden_states, new_states = self.model(
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
            conv_state_mask=conv_state_mask,
            recurrent_state_mask=recurrent_state_mask,
            valid_mask=valid_mask,
            batch_idx=batch_idx,
            output_hidden_states=output_hidden_states,
        )

        if "prefill" in self.phase and query_position is not None:
            hidden_states = hidden_states[:, query_position.to(torch.int).unsqueeze(0)]

        logits = self.lm_head(hidden_states)
        return logits, all_hidden_states, new_states


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
        # valid/conv/recurrent state masks are the LAST graph inputs (get_input_info order): pop them off the
        # end (LIFO -> valid, recurrent, conv), let the base build the standard prefix + per-layer state block,
        # then split that state list into two containers and reattach the masks.
        args = list(args)
        has_linear = any(t == "linear_attention" for t in self.config.layer_types)
        # batch_idx is appended LAST for prefill only (get_input_info); pop it first.
        batch_idx = args.pop() if (has_linear and "prefill" in self.phase) else None
        valid_mask = args.pop() if has_linear else None
        recurrent_state_mask = args.pop() if has_linear else None
        conv_state_mask = args.pop() if has_linear else None
        base = list(super().prepare_forward_args(*args))
        past_key_values, past_states = self._split_layer_states(base[-2])  # split by layer type
        base[-2] = past_key_values
        base.insert(-1, past_states)  # keep past_states next to past_key_values, just before rotary_emb (last).
        return (*base, conv_state_mask, recurrent_state_mask, valid_mask, batch_idx)

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
            conv_state_mask,
            recurrent_state_mask,
            valid_mask,
            batch_idx,
        ) = self.prepare_forward_args(*args)

        logits, all_hidden_states, new_states = self.model(
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
            conv_state_mask=conv_state_mask,
            recurrent_state_mask=recurrent_state_mask,
            valid_mask=valid_mask,
            batch_idx=batch_idx,
            output_hidden_states=self.rbln_config.output_hidden_states,
        )

        # Linear-attention state updates are returned so the runtime can persist them on the host.
        if self.rbln_config.output_hidden_states:
            return (logits, *new_states, *all_hidden_states)
        return (logits, *new_states)


class Qwen3_5_LanguageModelWrapper(Qwen3_5_CausalLMWrapper):
    """The hybrid Qwen3.5 text backbone wired for the vision-language runtime.

    Reuses ``Qwen3_5_CausalLMWrapper``'s hybrid graph rewrite (``convert_to_rbln_class``, the
    ``get_rbln_*`` factories that emit GatedDeltaNet linear layers + gated full-attention layers, and the
    linear-state threading in ``Qwen3_5Model``). The only changes vs the text-only wrapper:

    - ``model.config`` is a ``Qwen3_5Config`` (vision + text); swap it to ``text_config`` for the parent
      ``DecoderOnlyWrapper`` initialization (which expects text attributes).
    - the language model is reached via ``model.get_decoder()`` (nested under the multimodal model).
    - ``position_embeds`` (precomputed mRoPE cos/sin) is an explicit graph input, passed to the model as the
      ``rotary_emb`` tensor; there is no inline ``RotaryEmbedding`` and no deepstack.
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
        # valid/conv/recurrent state masks are the LAST graph inputs (get_input_info order): pop them off the
        # end first so the standard front-popping + `past_states = args` below is unchanged. Linear layers only.
        has_linear = any(t == "linear_attention" for t in self.config.layer_types)
        # batch_idx is appended LAST for prefill only (get_input_info); pop it first.
        batch_idx = args.pop() if (has_linear and "prefill" in self.phase) else None
        valid_mask = args.pop() if has_linear else None
        recurrent_state_mask = args.pop() if has_linear else None
        conv_state_mask = args.pop() if has_linear else None
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

        # The remaining args are 2 state slots per layer (flat); pair them up, then split by layer type via the
        # shared helper: past_key_values = (key, value) for full_attention, past_states = (conv_state,
        # recurrent_state) for linear_attention (both full-length with None at the other type's indices).
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
            conv_state_mask,
            recurrent_state_mask,
            valid_mask,
            batch_idx,
        )
