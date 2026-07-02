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

- The GatedDeltaNet uses the *chunked* (parallel) delta rule for prefill and the *recurrent*
  delta rule for decode (seq=1). The HF chunked kernel does not lower as-is, so it is rewritten
  (``rbln_chunk_gated_delta_rule``) to compile on RBLN; both forms share a state layout, so prefill
  hands its ``recurrent_state`` straight to recurrent-decode.
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
from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

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


def rbln_chunk_gated_delta_rule(query, key, value, g, beta, eye, tril_incl, tril_strict, initial_state, num_iter=1):
    """Single-window (parallel) gated delta rule for RBLN PREFILL. Numerically identical to HF
    ``torch_chunk_gated_delta_rule`` (cos=1.0), with four lowering fixes (see project memo
    ``rbln-qwen35-deltanet-compile``):

    - intra-window triangular inverse via the fixed-shape log-depth identity
      ``(I - A)^{-1} = (I + A)(I + A^2)(I + A^4)...`` instead of the variable-slice forward-substitution
      loop (the ScatterInfo blocker, also an unroll);
    - mask / identity constants kept at the tensors' rank ``(1, 1, S, S)`` (else PadChannels emits a
      wrong-rank pad);
    - float lower-triangular masks instead of bool ``masked_fill`` / ``.tril()`` (``rtosa.where``
      rejects i1);
    - matmuls kept to 2 batch dims ``(B, Hv)`` — the standard multi-head layout (the 3-batch-dim form
      trips OpTiling: "memory size mismatch (3 vs 4)").

    NO internal sub-chunking: the optimum-rbln prefill runtime already splits the prompt into fixed
    ``prefill_chunk_size`` windows and carries ``recurrent_state`` across them, so ONE call == ONE
    delta-rule chunk. ``query_length == prefill_chunk_size == S == the mask size``; the runtime pads
    the final window, so there is no padding here. The intra-window term is the causal attention over
    the window; the inter term is the interaction with the incoming ``initial_state`` (carried state).

    Inputs follow the recurrent rule's call site: query/key ``(B, S, Hv, Dk)``, value ``(B, S, Hv, Dv)``,
    g/beta ``(B, S, Hv)``, initial_state ``(B, Hv, Dk, Dv)``. Returns core ``(B, S, Hv, Dv)`` and the
    final state ``(B, Hv, Dk, Dv)`` — the same layout the recurrent rule returns. Used for PREFILL;
    decode (seq=1) stays on the recurrent rule.
    """
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    # (B, S, Hv, *) -> (B, Hv, S, *); one prefill window == one delta chunk (no internal split)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    Dk = key.shape[-1]

    query = query * (Dk**-0.5)
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    incr = g  # (B, Hv, S) per-token log-decay increments (pre-cumsum)
    g = incr.cumsum(dim=-1)  # (B, Hv, S) cumulative log-decay across the window

    diff = g.unsqueeze(-1) - g.unsqueeze(-2)  # (B, Hv, S, S)
    decay_mask = (diff * tril_incl).exp() * tril_incl
    A = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * tril_strict

    # (I - A)^{-1} = (I+A)(I+A^2)(I+A^4)...; ceil(log2(S)) squarings suffice since A^S = 0 (nilpotent).
    # S is the static prefill window size, so this trip count is a Python constant -> unrolls cleanly.
    M = eye + A
    P = A
    for _ in range(num_iter):  # (I - A)^{-1} via log-depth squaring
        P = P @ P
        M = M @ (eye + P)
    attn = M

    w = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_state = initial_state.to(w)  # (B, Hv, Dk, Dv) carried recurrent_state (already masked at the GDN entry)

    attn_intra = (query @ key.transpose(-1, -2)) * decay_mask
    v_new = w - k_cumdecay @ last_state
    attn_inter = (query * g.exp().unsqueeze(-1)) @ last_state
    core = attn_inter + attn_intra @ v_new  # (B, Hv, S, Dv)

    # decay from each position to the window end = g_total - g, as an exclusive REVERSE cumsum of the
    # increments. Avoids g[..., -1] (an innermost width-1 StridedSlice the ViewOpSplit pass rejects) and
    # the (…, 1)-vs-(…, S) broadcast subtract (which trips PadLastDim). All terms stay (B, Hv, S).
    _ax = incr.dim() - 1  # positive axis: RBLN's reverse op requires a non-negative axis (flip(-1) fails)
    rev_incl = incr.flip(_ax).cumsum(dim=-1).flip(_ax)  # Σ_{i>=t} incr[i]
    decay_to_end = rev_incl - incr  # Σ_{i>t} incr[i]  == g_total - g  (<= 0, numerically stable)
    g_total = incr.sum(dim=-1, keepdim=True)  # (B, Hv, 1) total log-decay (a reduction, not a slice)
    new_state = (
        last_state * g_total.unsqueeze(-1).exp() + (key * decay_to_end.exp().unsqueeze(-1)).transpose(-1, -2) @ v_new
    )

    core = core.transpose(1, 2).contiguous().to(initial_dtype)  # (B, S, Hv, Dv)
    return core, new_state


def rbln_recurrent_gated_delta_rule_step(query, key, value, g, beta, initial_state):
    """Single-step (decode, seq=1) gated delta rule for RBLN. Numerically identical to HF
    ``torch_recurrent_gated_delta_rule`` at S=1 (cos=1.0), but written without the ops that do NOT
    lower on RBLN and silently corrupt the result on device (measured cos≈0.2):

    - the per-position ``core_attn_out[:, :, i] = ...`` index-assign (a ScatterInfo — the SAME blocker
      the prefill chunk was written to avoid);
    - the ``query[:, :, i]`` position indexing (a dynamic StridedSlice);
    - the two ``.sum(dim=-2)`` reductions over the NON-innermost ``Dk`` axis — rewritten as matmuls
      (which lower cleanly), matching the chunk's "2 batch dims (B, Hv) + matmul" style.

    Since S=1 the recurrence is a single step with no loop and no output scatter. The seq dim is
    dropped with a data-preserving ``reshape`` (not an index). Inputs/outputs match the chunk and the
    HF rule: query/key ``(B, 1, Hv, Dk)``, value ``(B, 1, Hv, Dv)``, g/beta ``(B, 1, Hv)``,
    initial_state ``(B, Hv, Dk, Dv)`` -> core ``(B, 1, Hv, Dv)``, new_state ``(B, Hv, Dk, Dv)``.
    """
    initial_dtype = query.dtype
    batch_size, _, num_v_heads, k_head_dim = query.shape
    v_head_dim = value.shape[-1]
    # drop the S=1 axis via reshape (no position indexing) to a (B, Hv, D) 2-batch-dim layout, THEN
    # normalize. NB: HF's l2norm does `(x*x).sum(dim=-1)`, but on RBLN a reduction over the innermost
    # axis returns ~0 when the second-to-last dim is small (the S=1 decode tensors, e.g. (1,1,2,64)) ->
    # rsqrt(eps)≈1000 -> the norm (and everything downstream) blows up. So compute ||x||² as a matmul
    # dot-product (a contraction, which DOES lower correctly at these sizes).
    q = query.reshape(batch_size, num_v_heads, k_head_dim).float()
    k = key.reshape(batch_size, num_v_heads, k_head_dim).float()
    v = value.reshape(batch_size, num_v_heads, v_head_dim).float()

    def _l2norm_dot(x):  # x: (B, Hv, D) -> unit-normalized over D, via matmul sum-of-squares
        ss = torch.matmul(x.unsqueeze(-2), x.unsqueeze(-1)).squeeze(-1)  # (B,Hv,1,D)@(B,Hv,D,1)->(B,Hv,1)
        return x * torch.rsqrt(ss + 1e-6)

    q_row = (_l2norm_dot(q) * (k_head_dim**-0.5)).unsqueeze(-2)  # (B, Hv, 1, Dk)
    k_row = _l2norm_dot(k).unsqueeze(-2)  # (B, Hv, 1, Dk)
    v_row = v.unsqueeze(-2)  # (B, Hv, 1, Dv)
    g_t = g.reshape(batch_size, num_v_heads, 1, 1).float().exp()  # (B, Hv, 1, 1)
    beta_t = beta.reshape(batch_size, num_v_heads, 1, 1).float()  # (B, Hv, 1, 1)

    state = initial_state.float() * g_t  # decay the carried state
    kv_mem = torch.matmul(k_row, state)  # (B, Hv, 1, Dk) @ (B, Hv, Dk, Dv) = (B, Hv, 1, Dv)
    delta = (v_row - kv_mem) * beta_t  # (B, Hv, 1, Dv)
    new_state = state + torch.matmul(k_row.transpose(-1, -2), delta)  # + (B,Hv,Dk,1)@(B,Hv,1,Dv)
    core = torch.matmul(q_row, new_state)  # (B, Hv, 1, Dk) @ (B, Hv, Dk, Dv) = (B, Hv, 1, Dv)
    core = core.reshape(batch_size, 1, num_v_heads, v_head_dim).to(initial_dtype)  # (B, 1, Hv, Dv)
    return core, new_state


class Qwen3_5GatedDeltaNet(nn.Module):
    """GatedDeltaNet token mixer for RBLN (functional: states in -> states out).

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

        # PREFILL path: one prefill window == one delta-rule chunk (the runtime splits the prompt
        # into prefill_chunk_size windows and carries recurrent_state across them). The window math
        # needs fixed S×S constants — float lower-triangular masks + identity at the tensor rank
        # (1, 1, S, S), S = prefill_chunk_size — so the rewrite lowers on RBLN (see
        # rbln_chunk_gated_delta_rule). register_buffer(persistent=False): graph constants, not weights.
        self.prefill_chunk_size = getattr(rbln_config, "prefill_chunk_size", 128)
        _ones = torch.ones(self.prefill_chunk_size, self.prefill_chunk_size)
        _shape = (1, 1, self.prefill_chunk_size, self.prefill_chunk_size)
        self.register_buffer(
            "_prefill_chunk_eye", torch.eye(self.prefill_chunk_size).reshape(_shape), persistent=False
        )
        self.register_buffer("_prefill_chunk_tril_incl", torch.tril(_ones).reshape(_shape), persistent=False)
        self.register_buffer(
            "_prefill_chunk_tril_strict", torch.tril(_ones, diagonal=-1).reshape(_shape), persistent=False
        )
        # Upper-triangular-inclusive constant for the conv_state reverse-cumsum (rev = triu @ valid_mask).
        # Size is (K-1)+S, NOT S: the new conv_state is the last K-1 VALID rows of the FULL conv input
        # [conv_state | mixed_qkv], so a window with < K-1 valid tokens must reach back into the prepended
        # conv_state. A distinct buffer (not a view of tril_incl) — the compiler's weight-reusability
        # check rejects the same constant used in two op contexts.
        _conv_total = self.prefill_chunk_size + self.conv_kernel_size - 1
        self.register_buffer("_prefill_conv_triu", torch.triu(torch.ones(_conv_total, _conv_total)), persistent=False)
        # log-depth (I - A)^{-1} needs ceil(log2(S)) squarings, where S = the intra-window matrix
        # size = prefill_chunk_size (A is S×S strictly-lower, so A^S = 0). NOT num_v_heads.
        self.num_iter = max(1, (self.prefill_chunk_size - 1).bit_length())  # ceil(log2(prefill_chunk_size))

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
        valid_mask: Optional[torch.Tensor] = None,
        conv_state_mask: Optional[torch.Tensor] = None,
        recurrent_state_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        k_1 = self.conv_kernel_size - 1
        prefill = "prefill" in self._phase and valid_mask is not None

        # Force the PREFILL FIRST CHUNK to start from zero carried state: the runtime feeds a zeros mask
        # for prefill window 0 and a ones mask for every later window (masks have the same shape as the
        # states, so this is a plain elementwise multiply; mask==1 is exact -> later chunks unchanged).
        # Only in prefill — decode always continues from the real carried state.
        if prefill:
            if conv_state_mask is not None:
                conv_state = conv_state * conv_state_mask
            if recurrent_state_mask is not None:
                recurrent_state = recurrent_state * recurrent_state_mask

        mixed_qkv = self.in_proj_qkv(hidden_states)  # (B, S, conv_dim)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # HF-style channel-first depthwise conv: transpose to (B, conv_dim, S) and prepend the cached
        # left-context on the TIME axis (dim=-1), exactly like HF's `hidden_states_new`. Then F.conv1d
        # with padding=0 (context already prepended) -> output is exactly S long, no trailing slice.
        x_cf = torch.cat([conv_state.transpose(1, 2), mixed_qkv.transpose(1, 2)], dim=-1)  # (B, conv_dim, (K-1)+S)
        # new conv_state = the last K-1 conv INPUTS. In PREFILL the window is right-padded to
        # prefill_chunk_size, so the last K-1 columns are padding (NONZERO via projection biases, hence
        # still wrong); select the last K-1 VALID rows of the (raw) mixed_qkv via a reverse-cumsum of
        # valid_mask -> one-hot -> matmul (compile-safe: arithmetic + matmul, no dynamic StridedSlice).
        # Decode (no padding) uses the plain last-K-1 tail (HF-style innermost slice on the time axis).
        if prefill:
            # new conv_state = the last K-1 VALID rows of the FULL conv input [conv_state | mixed_qkv].
            # The prepended conv_state (first K-1 cols of x_cf) is ALWAYS valid left-context, so extend
            # valid_mask with K-1 leading ones. rev[t] = # valid from t to end (via triu matmul, avoiding
            # cumsum/flip); the last K-1 valid have rev in {K-1,..,1}; a clamp one-hot selects them and a
            # matmul gathers those rows. Gathering from the FULL x (not just mixed_qkv) is what lets a
            # window with < K-1 valid tokens still pull the tail of the previous window's conv_state
            # (e.g. multi-window prefill whose last window has 1-2 valid tokens). Keep the time axis
            # INNERMOST through abs/clamp (a (B, T, 1) innermost-1 tensor returns garbage on device).
            total = k_1 + seq_len
            vm_ext = torch.cat([valid_mask.new_ones(batch_size, k_1, 1), valid_mask], dim=1)  # (B, (K-1)+S, 1)
            rev = torch.matmul(self._prefill_conv_triu, vm_ext).reshape(batch_size, 1, total)  # (B, 1, (K-1)+S)
            jv = (k_1 - torch.arange(k_1, device=hidden_states.device)).view(1, k_1, 1)  # (1, K-1, 1): [K-1,...,1]
            sel = (1.0 - (rev - jv).abs().to(mixed_qkv.dtype)).clamp(0.0, 1.0)  # (B, K-1, (K-1)+S)
            new_conv_state = torch.matmul(sel, x_cf.transpose(1, 2)).contiguous()  # @ (B,(K-1)+S,conv_dim)=(B,K-1,conv_dim)
        else:
            new_conv_state = x_cf[:, :, -k_1:].transpose(1, 2).contiguous()  # (B, K-1, conv_dim), HF-style
        conv_out = F.conv1d(x_cf, self.conv1d.weight, self.conv1d.bias, padding=0, groups=self.conv_dim)
        mixed_qkv = F.silu(conv_out.transpose(1, 2))  # (B, S, conv_dim)

        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if prefill:
            # PREFILL right-padding: padding tokens have nonzero q/k/v/g via the projection biases, which
            # would pollute the recurrent state (a sum over the window) and its decay. Zero them out.
            g = g * valid_mask.to(g.dtype)
            beta = beta * valid_mask.to(beta.dtype)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if "prefill" in self._phase:
            # One prefill window == one delta chunk (compiles on RBLN); the runtime carries
            # recurrent_state across windows and hands the final state to recurrent-decode.
            core_attn_out, new_recurrent_state = rbln_chunk_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                self._prefill_chunk_eye,
                self._prefill_chunk_tril_incl,
                self._prefill_chunk_tril_strict,
                recurrent_state,
                num_iter=self.num_iter,
            )
        else:
            # Decode (seq=1): the single-step recurrent rule, rewritten for RBLN (HF's version lowers to
            # garbage on device — cos≈0.2 — via its output scatter and small-tensor l2norm reduction).
            core_attn_out, new_recurrent_state = rbln_recurrent_gated_delta_rule_step(
                query, key, value, g, beta, recurrent_state
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
        valid_mask: Optional[torch.Tensor] = None,
        conv_state_mask: Optional[torch.Tensor] = None,
        recurrent_state_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_conv_state, new_recurrent_state = self.linear_attn(
            hidden_states,
            conv_state,
            recurrent_state,
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
        conv_state_mask: Optional[torch.Tensor] = None,
        recurrent_state_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.get_embedding()(input_ids)
        hidden_states = inputs_embeds * self.hidden_multiplier

        # Per-token validity for the linear layers: PREFILL right-pads the window to prefill_chunk_size
        # with ZERO embeddings (F.pad in the runtime), so a valid token has nonzero inputs_embeds and a
        # padding token is exactly zero. Derive a (B, S, 1) 0/1 mask from that (a dynamic tensor -> not
        # constant-folded, unlike a scalar query_position). The linear layers use it to drop padding from
        # the recurrent-state sum/decay and the conv_state extraction; full windows -> all ones (no-op).
        valid_mask = (inputs_embeds.abs().sum(dim=-1, keepdim=True) * 1e9).clamp(0.0, 1.0)

        position_ids = position_ids if position_ids is not None else cache_position
        cos = sin = None
        if rotary_emb is not None:
            if isinstance(rotary_emb, torch.Tensor):
                # multimodal path: mRoPE cos/sin are precomputed on the host and passed in as a
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
                hidden_states, new_conv_state, new_recurrent_state = layer(
                    hidden_states,
                    conv_state,
                    recurrent_state,
                    valid_mask=valid_mask,
                    conv_state_mask=conv_state_mask,
                    recurrent_state_mask=recurrent_state_mask,
                )
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
        conv_state_mask: Optional[torch.Tensor] = None,
        recurrent_state_mask: Optional[torch.Tensor] = None,
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
            conv_state_mask=conv_state_mask,
            recurrent_state_mask=recurrent_state_mask,
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

    def prepare_forward_args(self, *args):
        # conv_state_mask / recurrent_state_mask are the LAST two graph inputs (see get_input_info) —
        # pop them off the end, let the base handle the standard prefix + per-layer state block, then
        # tack the masks back on. Only present when the model has linear_attention layers.
        args = list(args)
        has_linear = bool(getattr(self.rbln_config, "linear_attention_layers", None))
        recurrent_state_mask = args.pop() if has_linear else None
        conv_state_mask = args.pop() if has_linear else None
        base = super().prepare_forward_args(*args)
        return (*base, conv_state_mask, recurrent_state_mask)

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
            conv_state_mask,
            recurrent_state_mask,
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
            conv_state_mask=conv_state_mask,
            recurrent_state_mask=recurrent_state_mask,
            output_hidden_states=self.rbln_config.output_hidden_states,
        )

        # Linear-attention state updates are returned so the runtime can persist them on the host.
        if self.rbln_config.output_hidden_states:
            return (logits, *new_states, *all_hidden_states)
        return (logits, *new_states)


class Qwen3_5_LanguageModelWrapper(Qwen3_5_CausalLMWrapper):
    """The hybrid Qwen3.5 text backbone wired for the vision-language runtime.

    Reuses ``Qwen3_5Wrapper``'s hybrid graph rewrite (``convert_to_rbln_class``, the
    ``get_rbln_*`` factories that emit GatedDeltaNet linear layers + gated full-attention
    layers, and the linear-state threading in ``Qwen3_5Model``). The only changes vs the
    text-only wrapper:

    - ``model.config`` is a ``Qwen3_5Config`` (vision + text); swap it to ``text_config`` for
      the parent ``DecoderOnlyWrapper`` initialization (which expects text attributes).
    - the language model is reached via ``model.get_decoder()`` (it is nested under the multimodal model).
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
        # conv_state_mask / recurrent_state_mask are the LAST two graph inputs (see get_input_info):
        # pop them off the end first, so the standard front-popping + `past_key_values = args` below is
        # unchanged. Present only when the model has linear_attention layers.
        has_linear = bool(getattr(self.rbln_config, "linear_attention_layers", None))
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
            conv_state_mask,
            recurrent_state_mask,
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
            conv_state_mask,
            recurrent_state_mask,
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
            conv_state_mask=conv_state_mask,
            recurrent_state_mask=recurrent_state_mask,
            output_hidden_states=self.rbln_config.output_hidden_states,
        )

        # Linear-attention state updates are returned so the runtime can persist them.
        if self.rbln_config.output_hidden_states:
            return (logits, *new_states, *all_hidden_states)
        return (logits, *new_states)
