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

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..qwen3.qwen3_architecture import Qwen3Wrapper


if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASREncoderConfig

    from ....configuration_utils import RBLNModelConfig


# 104 tokens for the released model.
def get_window_size(config: "Qwen3ASREncoderConfig") -> int:
    return config.max_position_embeddings * (config.n_window_infer // (config.n_window * 2))


# Bidirectional attention over one window. HF's windows are variable-length (`torch.split(...,
# lengths.tolist())`), so every key is real and it attends with `attention_mask=None`. A static
# graph fixes the window size, audio length does not divide into it, and `attn_bias` masks the tail.
class Qwen3ASRAudioAttention(nn.Module):
    def __init__(self, model: nn.Module, rbln_config: "RBLNModelConfig"):
        super().__init__()
        self.num_heads = model.num_heads
        self.head_dim = model.head_dim
        self.q_proj = model.q_proj
        self.k_proj = model.k_proj
        self.v_proj = model.v_proj
        self.out_proj = model.out_proj
        self.scale = torch.tensor(model.scaling, dtype=rbln_config.dtype)

    def forward(self, hidden_states: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        num_windows, window_size, _ = hidden_states.shape
        shape = (num_windows, window_size, self.num_heads, self.head_dim)
        q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn_weights = attn_weights + attn_bias
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(num_windows, window_size, -1)
        return self.out_proj(attn_output)


class Qwen3ASRAudioEncoderLayer(nn.Module):
    def __init__(self, model: nn.Module, rbln_config: "RBLNModelConfig"):
        super().__init__()
        self.self_attn = Qwen3ASRAudioAttention(model.self_attn, rbln_config)
        self.self_attn_layer_norm = model.self_attn_layer_norm
        self.activation_fn = model.activation_fn
        self.fc1 = model.fc1
        self.fc2 = model.fc2
        self.final_layer_norm = model.final_layer_norm

    def forward(self, hidden_states: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.self_attn_layer_norm(hidden_states), attn_bias)
        residual = self.final_layer_norm(hidden_states)
        residual = self.fc2(self.activation_fn(self.fc1(residual)))
        hidden_states = hidden_states + residual

        # HF's overflow guard; the dtype is fixed at trace time, so it folds away elsewhere.
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


# HF packs the valid post-convolution tokens with a `nonzero()`, whose output length follows the
# mask's contents rather than its shape. Every chunk keeps its padding here instead, so windows
# are a plain reshape and the host drops the invalid rows from the projector's output.
class Qwen3ASREncoderWrapper(nn.Module):
    def __init__(self, model: nn.Module, rbln_config: "RBLNModelConfig"):
        super().__init__()
        self.conv2d1 = model.conv2d1
        self.conv2d2 = model.conv2d2
        self.conv2d3 = model.conv2d3
        self.conv_out = model.conv_out
        self.positional_embedding = model.positional_embedding
        self.layers = nn.ModuleList(Qwen3ASRAudioEncoderLayer(layer, rbln_config) for layer in model.layers)
        self.ln_post = model.ln_post
        if not hasattr(model, "multi_modal_projector"):
            raise AttributeError(
                "The audio tower is missing `multi_modal_projector`. It is attached by "
                "`RBLNQwen3ASRForConditionalGeneration._reconstruct_model_if_needed`; export the "
                "encoder through `RBLNQwen3ASRForConditionalGeneration` rather than on its own."
            )
        self.multi_modal_projector = model.multi_modal_projector

    # input_features: `(num_windows * chunks_per_window, 1, num_mel_bins, chunk_len)` log-mel chunks.
    # attn_mask: `(num_windows, 1, 1, window_size)`, 1 for valid keys.
    def forward(self, input_features: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        num_windows = attn_mask.shape[0]

        conv_out = nn.functional.gelu(self.conv2d1(input_features))
        conv_out = nn.functional.gelu(self.conv2d2(conv_out))
        conv_out = nn.functional.gelu(self.conv2d3(conv_out))
        total_chunks, conv_channels, freq_bins, time_steps = conv_out.shape
        conv_out = self.conv_out(
            conv_out.permute(0, 3, 1, 2).contiguous().view(total_chunks, time_steps, conv_channels * freq_bins)
        )
        conv_out += self.positional_embedding.positional_embedding[:time_steps].to(conv_out.dtype)

        hidden_states = conv_out.view(num_windows, -1, conv_out.shape[-1])

        attn_bias = (1.0 - attn_mask) * torch.finfo(hidden_states.dtype).min
        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_bias)
        return self.multi_modal_projector(self.ln_post(hidden_states))


# The base looks for `model.model.layers`; here the decoder is nested one level deeper.
class Qwen3ASRLanguageModelWrapper(Qwen3Wrapper):
    def get_decoder_layers(self, model: "PreTrainedModel"):
        return model.get_decoder().layers

    def get_model_layer(self, model: "PreTrainedModel"):
        return model.get_decoder()
