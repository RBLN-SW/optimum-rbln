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
    "rbln_custom_ops::paged_attn_decode",
    mutates_args=(["kcache", "vcache"]),
)
def paged_attn_decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
) -> Tensor:
    return torch.empty_like(q)


@paged_attn_decode.register_fake
def paged_attn_decode_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_attn_decode_kv_fp8",
    mutates_args=(["kcache", "vcache"]),
)
def paged_attn_decode_kv_fp8(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    k_scale: Tensor,
    v_scale: Tensor,
) -> Tensor:
    return torch.empty_like(q)


@paged_attn_decode_kv_fp8.register_fake
def paged_attn_decode_kv_fp8_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    k_scale: Tensor,
    v_scale: Tensor,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_attn_prefill",
    mutates_args=(["kcache", "vcache"]),
)
def paged_attn_prefill(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
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
    - mask: [batch=1, 1, 1, seq_len, max_seq_len] - Attention mask
    - kcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Key cache
    - vcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Value cache
    - seq: [1, 1] - Starting sequence position
    - scale: [] - Attention scale factor
    - block_table: [batch_size, max_seq_len // block_size] - Block indices for KV cache management
    - block_size: [] - Number of tokens per block

    Returns:
        Tensor: attn_output: [batch=1, n_heads, n_groups, seq_len, head_dim] - Attention output
    """
    return torch.empty_like(q)


@paged_attn_prefill.register_fake
def paged_attn_prefill_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_attn_prefill_kv_fp8",
    mutates_args=(["kcache", "vcache"]),
)
def paged_attn_prefill_kv_fp8(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    k_scale: Tensor,
    v_scale: Tensor,
) -> Tensor:
    return torch.empty_like(q)


@paged_attn_prefill_kv_fp8.register_fake
def paged_attn_prefill_kv_fp8_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    k_scale: Tensor,
    v_scale: Tensor,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_causal_attn_decode",
    mutates_args=(["kcache", "vcache"]),
)
def paged_causal_attn_decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    mask: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    """Defines the computation pattern for fused attention with KV cache updates.

    IMPORTANT: This op serves as a pattern definition for the RBLN compiler to generate
    a single optimized NPU operation. It is NOT meant for CPU execution.

    Pattern components that compiler fuses into a single op:
    1. KV cache updates with new key/value states
    2. Scaled dot-product attention computation
    3. Causal masked softmax operation
    4. Final attention output computation

    Expected tensor shapes:
    - q: [batch=1, n_heads, n_groups, 1, head_dim] - Query states for single token
    - k: [batch=1, n_heads, 1, 1, head_dim] - Key states for current input
    - v: [batch=1, n_heads, 1, 1, head_dim] - Value states for current input
    - kcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Key cache
    - vcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Value cache
    - seq: [1, 1] - Starting sequence position
    - scale: [] - Attention scale factor
    - block_table: [batch_size, max_seq_len // block_size] - Block indices for KV cache management
    - block_size: [] - Number of tokens per block
    - mask: [batch=1, max_seq_len] - attention mask when use position_ids
    - s_aux: [num_attention_heads, sink_len] - auxiliary states for attention

    Returns:
        Tensor: attn_output: [batch=1, n_heads, n_groups, 1, head_dim] - Attention output
    """
    return torch.empty_like(q)


@paged_causal_attn_decode.register_fake
def paged_causal_attn_decode_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    mask: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_causal_attn_prefill",
    mutates_args=(["kcache", "vcache"]),
)
def paged_causal_attn_prefill(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    is_bidirectional: bool,
    mask: Optional[Tensor] = None,
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
    - batch: [1] - Batch index for cache access
    - seq: [1, 1] - Starting sequence position
    - scale: [] - Attention scale factor
    - block_table: [batch_size, max_seq_len // block_size] - Block indices for KV cache management
    - block_size: [] - Number of tokens per block
    - is_bidirectional: [] - Whether the attention is bidirectional at current sequence position
    - mask: [batch=1, max_seq_len] - attention mask when use position_ids
    - s_aux: [num_attention_heads, sink_len] - auxiliary states for attention

    Returns:
        Tensor: attn_output: [batch=1, n_heads, n_groups, seq_len, head_dim] - Attention output
    """
    return torch.empty_like(q)


@paged_causal_attn_prefill.register_fake
def paged_causal_attn_prefill_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    is_bidirectional: bool,
    mask: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_causal_attn_decode_kv_fp8",
    mutates_args=(["kcache", "vcache"]),
)
def paged_causal_attn_decode_kv_fp8(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    k_scale: Tensor,
    v_scale: Tensor,
    mask: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(q)


@paged_causal_attn_decode_kv_fp8.register_fake
def paged_causal_attn_decode_kv_fp8_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    k_scale: Tensor,
    v_scale: Tensor,
    mask: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_causal_attn_prefill_kv_fp8",
    mutates_args=(["kcache", "vcache"]),
)
def paged_causal_attn_prefill_kv_fp8(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    is_bidirectional: bool,
    k_scale: Tensor,
    v_scale: Tensor,
    mask: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(q)


@paged_causal_attn_prefill_kv_fp8.register_fake
def paged_causal_attn_prefill_kv_fp8_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    is_bidirectional: bool,
    k_scale: Tensor,
    v_scale: Tensor,
    mask: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_add_softmax_attn_decode",
    mutates_args=(["kcache", "vcache"]),
)
def paged_add_softmax_attn_decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
) -> Tensor:
    """Defines the computation pattern for fused attention with KV cache updates.

    IMPORTANT: This op serves as a pattern definition for the RBLN compiler to generate
    a single optimized NPU operation. It is NOT meant for CPU execution.

    Pattern components that compiler fuses into a single op:
    1. KV cache updates with new key/value states
    2. Scaled dot-product attention computation
    3. add-softmax operation
    4. Final attention output computation

    Expected tensor shapes:
    - q: [batch=1, n_heads, n_groups, 1, head_dim] - Query states for single token
    - k: [batch=1, n_heads, 1, 1, head_dim] - Key states for current input
    - v: [batch=1, n_heads, 1, 1, head_dim] - Value states for current input
    - mask: [batch=1, n_heads, 1, 1, max_seq_len] - Attention mask
    - kcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Key cache
    - vcache: [batch_size, n_heads, 1, max_seq_len, head_dim] - Value cache
    - seq: [1] - Current sequence position
    - scale: [] - Attention scale factor
    - block_table: [batch_size, max_seq_len // block_size] - Block indices for KV cache management
    - block_size: [] - Number of tokens per block

    Returns:
        Tensor: attn_output: [batch=1, n_heads, 1, 1, head_dim] - Attention output
    """
    return torch.empty_like(q)


@paged_add_softmax_attn_decode.register_fake
def paged_add_softmax_attn_decode_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
) -> Tensor:
    return torch.empty_like(q)
