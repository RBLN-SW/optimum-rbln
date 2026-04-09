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


from typing import Optional

import torch
from torch import Tensor


@torch.library.custom_op(
    "rbln_custom_ops::paged_sliding_window_attn_prefill",
    mutates_args=(["kcache", "vcache"]),
)
def paged_sliding_window_attn_prefill(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    cache_seq_len: Tensor,
    cache_offset: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    is_bidirectional: bool,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    """Defines the computation pattern for prefill phase attention with KV cache updates.

    IMPORTANT: This op serves as a pattern definition for the RBLN compiler to generate
    a single optimized NPU operation. It is NOT meant for CPU execution.

    Key differences from decode pattern:
    - Handles prefill phase with multiple input tokens
    - Takes explicit batch index for continuous batching

    Expected tensor shapes:
    - q: [batch=1, n_heads, n_groups, seq_len, head_dim] - Query states for multiple tokens
    - k: [batch=1, n_heads, 1, seq_len, head_dim] - Key states for current input
    - v: [batch=1, n_heads, 1, seq_len, head_dim] - Value states for current input
    - kcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Key cache
    - vcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Value cache
    - cache_seq_len: [] - the sequence length of the cached states that were seen by the model
    - cache_offset: [] - The valid length in the combined sequence of the KV cache and the current projected key states.
    - scale: [] - Attention scale factor
    - is_bidirectional: [] - Whether the attention is bidirectional
    - s_aux: [num_attention_heads, sink_len] - auxiliary states for attention
    Returns:
        Tensor: attn_output: [batch=1, n_heads, n_groups, seq_len, head_dim] - Attention output
    """
    return torch.empty_like(q)


@paged_sliding_window_attn_prefill.register_fake
def paged_sliding_window_attn_prefill_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    cache_seq_len: Tensor,
    cache_offset: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    is_bidirectional: bool,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_sliding_window_attn_decode",
    mutates_args=(["kcache", "vcache"]),
)
def paged_sliding_window_attn_decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    cache_seq_len: Tensor,
    cache_offset: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    attn_mask: Tensor,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(q)


@paged_sliding_window_attn_decode.register_fake
def paged_sliding_window_attn_decode_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    cache_seq_len: Tensor,
    cache_offset: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    attn_mask: Tensor,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(q)
