"""The GatedDeltaNet core in plain torch, as the golden for the custom-op core.

Lifted from `qwen3_5_architecture` when that module dropped its native core: every configuration now
runs as `rbln_custom_ops::chunk_gated_delta_rule` / `recurrent_gated_delta_rule`. The numerics are
unchanged, so a test that compares the two still compares the custom op against the delta rule it
implements.
"""

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

from optimum.rbln.transformers.models.qwen3_5.qwen3_5_architecture import Qwen3_5GatedDeltaNet


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
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
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
    ).to(torch.float32)

    # inter-chunk: sequential carry across sub-chunks.
    core_chunks = []
    for i in range(n_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_intra = (q_i @ k_i.transpose(-1, -2)) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i].exp().unsqueeze(-1)) @ last_recurrent_state
        core_chunks.append(attn_inter + attn_intra @ v_new)

        incr_i = incr[:, :, i]
        g_total = incr_i.sum(dim=-1, keepdim=True)
        decay_to_end = incr_i @ tril_strict[0, 0, 0].to(incr_i.dtype)
        last_recurrent_state = (
            last_recurrent_state * g_total.unsqueeze(-1).exp()
            + (k_i * decay_to_end.exp().unsqueeze(-1)).transpose(-1, -2) @ v_new
        )

    # concat sub-chunk outputs along seq
    core = torch.cat(core_chunks, dim=2)
    core = core.transpose(1, 2).contiguous().to(initial_dtype)
    # return the final state in the 3D cache layout (B, Hv*Dk, Dv) to match the static cache.
    last_recurrent_state = last_recurrent_state.reshape(
        last_recurrent_state.shape[0],
        last_recurrent_state.shape[1] * last_recurrent_state.shape[2],
        last_recurrent_state.shape[3],
    ).to(initial_dtype)
    return core, last_recurrent_state


def rbln_recurrent_gated_delta_rule_step(query, key, value, g, beta, initial_state, use_qk_l2norm_in_kernel=False):
    """Single-step (decode, seq=1) gated delta rule for RBLN. Numerically identical to HF
    ``torch_recurrent_gated_delta_rule`` at S=1 (reimplemented to work around RBLN lowering; see inline notes).
    Inputs query/key ``(B,1,Hv,Dk)``, value ``(B,1,Hv,Dv)``, g/beta ``(B,1,Hv)``, initial_state the 3D cache
    layout ``(B,Hv,Dk*Dv)`` -> core ``(B,1,Hv,Dv)``, new_state ``(B,Hv,Dk*Dv)``. The state is kept 3D in the
    cache and reshaped to 4D here only after a compute (``* g_t``).
    """
    initial_dtype = query.dtype
    query, key, value, g, beta = [x.to(torch.float32) for x in (query, key, value, g, beta)]
    initial_state = initial_state.to(torch.float32)
    batch_size, _, num_v_heads, k_head_dim = query.shape
    v_head_dim = value.shape[-1]
    q = query.reshape(batch_size, num_v_heads, k_head_dim)
    k = key.reshape(batch_size, num_v_heads, k_head_dim)
    v = value.reshape(batch_size, num_v_heads, v_head_dim)

    def _l2norm_dot(x):
        # Same as HF l2norm, but ||x||² via matmul dot-product instead of `(x*x).sum(-1)`: on the tiny seq=1
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


class NativeGatedDeltaNetReference(Qwen3_5GatedDeltaNet):
    """The delta rule in plain torch. Serves every configuration the custom ops do not: a REBEL target,
    a `gdn_chunk_size` other than 128, and an artifact compiled before `gdn_custom_kernel`."""

    def __init__(self, linear_attn, rbln_config, layer_idx: int):
        super().__init__(linear_attn, rbln_config, layer_idx)
        # The custom core does not need these; the chunked delta rule below does.
        self.head_v_dim = linear_attn.head_v_dim
        self.prefill_chunk_size = rbln_config.prefill_chunk_size
        self.chunk_size = rbln_config.gdn_chunk_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        query_position: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        slot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        prefill = "prefill" in self._phase and valid_mask is not None
        # PREFILL window 0 starts from zero carried state: the runtime feeds a zeros mask for window 0 and a
        # ones mask afterward. Decode always carries.
        if prefill:
            if state_mask is not None:
                conv_state = conv_state * state_mask.reshape(-1, 1, 1)
                recurrent_state = recurrent_state * state_mask.reshape(-1, 1, 1)
        batch_size, seq_len, _ = hidden_states.shape
        k_1 = self.conv_kernel_size - 1

        batch_size, seq_len, _ = hidden_states.shape
        k_1 = self.conv_kernel_size - 1

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
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
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
            # fp32 like HF torch_chunk_gated_delta_rule, so `incr @ tril` stays single-dtype for the fp32 kernel.
            chunk_tril_incl = torch.tril(torch.ones(_cshape, device=query.device, dtype=torch.float32), diagonal=0)
            chunk_tril_strict = torch.tril(torch.ones(_cshape, device=query.device, dtype=torch.float32), diagonal=-1)
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
