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
    "rbln_custom_ops::paged_flash_attn_decode",
    mutates_args=(["kcache", "vcache"]),
)
def paged_flash_attn_decode(
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
    partition: int,
) -> Tensor:
    """Defines the computation pattern for fused flash attention with KV cache for decoding.

    Returns a tensor with the same shape as q.
    """
    return torch.empty_like(q)


@paged_flash_attn_decode.register_fake
def paged_flash_attn_decode_fake(
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
    partition: int,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_flash_attn_decode_kv_fp8",
    mutates_args=(["kcache", "vcache"]),
)
def paged_flash_attn_decode_kv_fp8(
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
    partition: int,
    k_scale: Tensor,
    v_scale: Tensor,
) -> Tensor:
    return torch.empty_like(q)


@paged_flash_attn_decode_kv_fp8.register_fake
def paged_flash_attn_decode_kv_fp8_fake(
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
    partition: int,
    k_scale: Tensor,
    v_scale: Tensor,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_flash_attn_prefill",
    mutates_args=(["kcache", "vcache"]),
)
def paged_flash_attn_prefill(
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
    partition: int,
) -> Tensor:
    """Defines the computation pattern for fused flash attention with KV cache for prefill.

    Returns a tensor with the same shape as q.
    """
    return torch.empty_like(q)


@paged_flash_attn_prefill.register_fake
def paged_flash_attn_prefill_fake(
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
    partition: int,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_flash_attn_prefill_kv_fp8",
    mutates_args=(["kcache", "vcache"]),
)
def paged_flash_attn_prefill_kv_fp8(
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
    partition: int,
    k_scale: Tensor,
    v_scale: Tensor,
) -> Tensor:
    return torch.empty_like(q)


@paged_flash_attn_prefill_kv_fp8.register_fake
def paged_flash_attn_prefill_kv_fp8_fake(
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
    partition: int,
    k_scale: Tensor,
    v_scale: Tensor,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_flash_causal_attn_decode",
    mutates_args=(["kcache", "vcache"]),
)
def paged_flash_causal_attn_decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    partition: int,
    mask: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    """Defines the computation pattern for fused causal flash attention with KV cache for decoding.

    Returns a tensor with the same shape as q.
    """
    return torch.empty_like(q)


@paged_flash_causal_attn_decode.register_fake
def paged_flash_causal_attn_decode_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    partition: int,
    mask: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_flash_causal_attn_decode_kv_fp8",
    mutates_args=(["kcache", "vcache"]),
)
def paged_flash_causal_attn_decode_kv_fp8(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    partition: int,
    k_scale: Tensor,
    v_scale: Tensor,
    mask: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(q)


@paged_flash_causal_attn_decode_kv_fp8.register_fake
def paged_flash_causal_attn_decode_kv_fp8_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    partition: int,
    k_scale: Tensor,
    v_scale: Tensor,
    mask: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_flash_causal_attn_prefill",
    mutates_args=(["kcache", "vcache"]),
)
def paged_flash_causal_attn_prefill(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    partition: int,
    is_bidirectional: bool,
    mask: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    """Defines the computation pattern for fused causal flash attention with KV cache for prefill.

    Returns a tensor with the same shape as q.
    """
    return torch.empty_like(q)


@paged_flash_causal_attn_prefill.register_fake
def paged_flash_causal_attn_prefill_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    partition: int,
    is_bidirectional: bool,
    mask: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::paged_flash_causal_attn_prefill_kv_fp8",
    mutates_args=(["kcache", "vcache"]),
)
def paged_flash_causal_attn_prefill_kv_fp8(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    partition: int,
    is_bidirectional: bool,
    k_scale: Tensor,
    v_scale: Tensor,
    mask: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(q)


@paged_flash_causal_attn_prefill_kv_fp8.register_fake
def paged_flash_causal_attn_prefill_kv_fp8_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kcache: Tensor,
    vcache: Tensor,
    seq: Tensor,
    scale: Tensor,
    block_table: Tensor,
    block_size: int,
    partition: int,
    is_bidirectional: bool,
    k_scale: Tensor,
    v_scale: Tensor,
    mask: Optional[Tensor] = None,
    s_aux: Optional[Tensor] = None,
) -> Tensor:
    return torch.empty_like(q)
