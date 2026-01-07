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
from typing import Tuple

import torch
from torch import nn
from transformers import PreTrainedModel


class Mamba2StepWrapper(nn.Module):
    """
    A fully-unrolled, branch-free **single-token** forward for HF Mamba2 that:
      - Avoids HF python branching on `cache_position`
      - Avoids full-seq (chunked scan) codepaths
      - Updates `conv_states` / `ssm_states` in-place

    This is the "wrap sub-classes" strategy: we do not call HF `Mamba2Mixer.forward`,
    we reuse its parameters/modules but implement the step math directly.
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model
        self.config = model.config

        # HF layout
        self.backbone = model.backbone
        self.embeddings = self.backbone.embeddings
        self.layers = self.backbone.layers
        self.norm_f = self.backbone.norm_f
        self.lm_head = model.lm_head

    @staticmethod
    def _finite_time_step_max(tmax: float) -> float:
        if math.isfinite(tmax):
            return float(tmax)
        # Avoid inf constants in compiler paths
        return 1.0e4

    def _mixer_step(
        self,
        mixer: nn.Module,
        hidden_states: torch.Tensor,  # [B,1,H]
        conv_state: torch.Tensor,  # [B, conv_dim, K]
        ssm_state: torch.Tensor,  # [B, num_heads, head_dim, state_size]
    ) -> torch.Tensor:
        # 1) in_proj: projection_size = intermediate + conv_dim + num_heads (d_mlp == 0)
        projected = mixer.in_proj(hidden_states)  # [B,1,proj]
        gate, hidden_states_B_C, dt = projected.split(
            [mixer.intermediate_size, mixer.conv_dim, mixer.num_heads], dim=-1
        )

        # 2) depthwise causal conv step:
        # update conv_state = shift-left + append new input at the end
        new_in = hidden_states_B_C.squeeze(1)  # [B, conv_dim]
        new_conv_state = torch.cat([conv_state[:, :, 1:], new_in.unsqueeze(-1)], dim=-1)
        conv_state.copy_(new_conv_state)

        # conv_out = sum_k conv_state[...,k] * weight[...,k] (+ bias)
        w = mixer.conv1d.weight.squeeze(1)  # [conv_dim, K]
        conv_out = torch.sum(new_conv_state * w.unsqueeze(0), dim=-1)  # [B, conv_dim]
        if getattr(mixer, "use_conv_bias", False) and mixer.conv1d.bias is not None:
            conv_out = conv_out + mixer.conv1d.bias
        conv_out = mixer.act(conv_out)  # [B, conv_dim]

        # Split conv_out into x, B, C
        x_vec, B_vec, C_vec = torch.split(
            conv_out,
            [mixer.intermediate_size, mixer.n_groups * mixer.ssm_state_size, mixer.n_groups * mixer.ssm_state_size],
            dim=-1,
        )

        # 3) SSM step update
        Bsz = x_vec.shape[0]
        x = x_vec.view(Bsz, mixer.num_heads, mixer.head_dim)  # [B, Hh, Dh]

        dt = dt.squeeze(1)  # [B, Hh]
        dt = dt[:, :, None].expand(-1, -1, mixer.head_dim)  # [B, Hh, Dh]
        dt_bias = mixer.dt_bias[None, :, None].expand(Bsz, mixer.num_heads, mixer.head_dim)
        dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
        tmin, tmax = mixer.time_step_limit
        dt = torch.clamp(dt, float(tmin), self._finite_time_step_max(float(tmax)))

        A = -torch.exp(mixer.A_log.float())  # [Hh]
        A = A[:, None, None].expand(mixer.num_heads, mixer.head_dim, mixer.ssm_state_size).to(dtype=torch.float32)
        dA = torch.exp(dt[..., None] * A[None, ...])  # [B, Hh, Dh, N]

        # B/C group expansion
        B = B_vec.view(Bsz, mixer.n_groups, mixer.ssm_state_size)
        C = C_vec.view(Bsz, mixer.n_groups, mixer.ssm_state_size)
        rep = mixer.num_heads // mixer.n_groups
        B = B.repeat_interleave(rep, dim=1)  # [B, Hh, N]
        C = C.repeat_interleave(rep, dim=1)  # [B, Hh, N]

        dB = dt[..., None] * B[:, :, None, :]  # [B, Hh, Dh, N]
        dBx = dB * x[..., None]  # [B, Hh, Dh, N]

        new_ssm = ssm_state * dA.to(ssm_state.dtype) + dBx.to(ssm_state.dtype)
        ssm_state.copy_(new_ssm)

        # y = <state, C>  (avoid bmm+reshape patterns that some compiler paths struggle with)
        # new_ssm: [B, Hh, Dh, N], C: [B, Hh, N]
        y = torch.sum(new_ssm * C[:, :, None, :].to(new_ssm.dtype), dim=-1)  # [B, Hh, Dh]

        D = mixer.D[..., None].expand(mixer.num_heads, mixer.head_dim).to(y.dtype)  # [Hh, Dh]
        y = y + x.to(y.dtype) * D[None, ...]
        y = y.reshape(Bsz, 1, mixer.num_heads * mixer.head_dim)  # [B,1,intermediate]

        # gated norm + out proj
        y = mixer.norm(y, gate)
        out = mixer.out_proj(y)
        return out, conv_state, ssm_state

    def forward(self, input_ids: torch.Tensor, conv_states: torch.Tensor, ssm_states: torch.Tensor) -> Tuple[torch.Tensor]:
        # input_ids: [B,1]
        hidden_states = self.embeddings(input_ids)  # [B,1,H]

        for layer_idx, block in enumerate(self.layers):
            residual = hidden_states
            hidden_states = block.norm(hidden_states.to(dtype=block.norm.weight.dtype))
            if getattr(block, "residual_in_fp32", False):
                residual = residual.to(torch.float32)

            conv_state = conv_states[layer_idx]
            ssm_state = ssm_states[layer_idx]
            hidden_states, conv_state, ssm_state = self._mixer_step(block.mixer, hidden_states, conv_state, ssm_state)
            conv_states[layer_idx] = conv_state
            ssm_states[layer_idx] = ssm_state
            hidden_states = residual + hidden_states

        hidden_states = self.norm_f(hidden_states)
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        return (logits, conv_states, ssm_states)