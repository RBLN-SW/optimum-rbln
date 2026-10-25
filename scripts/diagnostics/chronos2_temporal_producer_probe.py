"""Existing full encoder0 graph with true first-temporal producer observations."""

import torch
from chronos.chronos2.layers import Chronos2RotaryEmbedding
from einops import rearrange
from torch import nn


BASE_OUTPUT_NAMES = ("block0_time", "block0_group", "block0_output", "block1_output", "block2_output")
PRODUCER_NAMES = ("time0_norm", "time0_q_rope", "time0_k_rope", "time0_v", "time0_attention_context")
OUTPUT_NAMES = BASE_OUTPUT_NAMES + PRODUCER_NAMES


class TemporalProducerProbe(nn.Module):
    """Same three trained blocks, with native producers returned, never recomputed."""

    def __init__(self, layers):
        super().__init__()
        if len(layers) != 3:
            raise ValueError("Expected original first three complete encoder blocks")
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden, attention_mask, group_time_mask):
        positions = torch.arange(hidden.shape[1], dtype=torch.long, device=hidden.device).unsqueeze(0)
        first = self.layers[0]
        temporal = first.layer[0]
        mha = temporal.self_attention
        normed = temporal.layer_norm(hidden)

        def shape(value):
            return rearrange(value, "b s (h d) -> b h s d", h=mha.n_heads, s=hidden.shape[1], d=mha.kv_proj_dim)

        def unshape(value):
            return rearrange(value, "b h s d -> b s (h d)", h=mha.n_heads, s=hidden.shape[1], d=mha.kv_proj_dim)

        query = shape(mha.q(normed))
        key = shape(mha.k(normed))
        value = shape(mha.v(normed))
        cos, sin = mha.rope_embed(value, positions)
        query, key = Chronos2RotaryEmbedding.apply_rotary_pos_emb(query, key, cos, sin)
        if mha.config._attn_implementation == "sdpa":
            context, _ = mha._sdpa_attention(query, key, value, attention_mask)
        else:
            context, _ = mha._eager_attention(query, key, value, attention_mask)
        context = unshape(context)
        projected = mha.o(context)
        time0 = hidden + temporal.dropout(projected)
        group0 = first.layer[1](time0, attention_mask=group_time_mask, output_attentions=False)[0]
        block0 = first.layer[2](group0)
        block1 = self.layers[1](
            block0,
            position_ids=positions,
            attention_mask=attention_mask,
            group_time_mask=group_time_mask,
            output_attentions=False,
        ).hidden_states
        block2 = self.layers[2](
            block1,
            position_ids=positions,
            attention_mask=attention_mask,
            group_time_mask=group_time_mask,
            output_attentions=False,
        ).hidden_states

        # Observe six original rows only; all512 rows and allfive full boundaries remain computed.
        def selected(value):
            return torch.cat((value[:5], value[-1:]), dim=0)

        return (
            time0,
            group0,
            block0,
            block1,
            block2,
            selected(normed),
            selected(unshape(query)),
            selected(unshape(key)),
            selected(unshape(value)),
            selected(context),
        )
