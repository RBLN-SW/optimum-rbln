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

import math
from typing import Any, cast

from ....configuration_utils import RBLNModelConfig
from ....transformers.modeling_attention_utils import align


class RBLNQwenImage21Transformer2DModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN Qwen-Image 2.1 Transformer models.

    This class inherits from RBLNModelConfig and provides specific configuration options
    for the Transformer model of Qwen-Image 2.1.
    """

    def __init__(
        self,
        batch_size: int | None = None,
        sample_size: tuple[int, int] | list[tuple[int, int]] | None = None,
        max_seq_len: int | None = None,
        prefill_chunk_size: int | None = None,
        kvcache_num_prompts: int | None = None,
        attn_impl: str | None = None,
        kvcache_partition_len: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size for inference. Defaults to 1. Only 1 is supported.
            sample_size (tuple[int, int] | list[tuple[int, int]] | None): The latent (height, width) of an image
                block, in tokens, or a list of them: the target image sizes and, for image editing, the condition
                image sizes. Each distinct token count compiles one image graph; sizes with the same count (e.g. 4:3
                and 3:4) share it. An image block runs in the smallest graph that holds it, padded, so a size only
                needs a graph at least as large. Filled by the pipeline config; required for a standalone compile.
            max_seq_len (int | None): The maximum number of tokens before the target image: the prompt tokens and,
                for image editing, the condition-image latent tokens. Filled by the pipeline config; required for a
                standalone compile.
            prefill_chunk_size (int | None): The number of text tokens per causal chunk. Defaults to 128 (512 on
                RBLN-CR NPUs).
            kvcache_num_prompts (int | None): The number of prompts whose keys and values the KV cache holds at
                once. Classifier-free guidance (`true_cfg_scale > 1` with a negative prompt) uses two. Defaults to 1.
            attn_impl (str | None): The attention implementation, `"eager"` or `"flash_attn"`. Eager attention keeps
                each prompt's KV cache in one block; flash attention splits it into partitions of
                `kvcache_partition_len` tokens. Defaults to `"eager"`, or to `"flash_attn"` when
                `kvcache_partition_len` is set.
            kvcache_partition_len (int | None): The KV-cache partition length for flash attention. An image block
                runs as one chunk and a chunk cannot cross a partition, so it must hold the largest image block.
                Defaults to the NPU's default partition length when `attn_impl` is `"flash_attn"`.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        if sample_size is not None and isinstance(sample_size[0], int):
            sample_size = [sample_size]
        self.sample_size = None if sample_size is None else [(size[0], size[1]) for size in sample_size]
        self.max_seq_len = max_seq_len
        self.prefill_chunk_size = prefill_chunk_size
        self.kvcache_num_prompts = kvcache_num_prompts
        self.attn_impl = attn_impl
        self.kvcache_partition_len = kvcache_partition_len

    @property
    def image_prefill_chunk_sizes(self) -> list[int]:
        # An image graph depends only on the token count, as the host computes the RoPE tables.
        return sorted({math.prod(size) for size in cast(list[tuple[int, int]], self.sample_size)})

    @property
    def kvcache_len(self) -> int:
        # KV-cache slots per prompt: the prefix ends by `max_seq_len`, and the chunk after it writes its whole size.
        # `_update_rbln_config` sets the lengths before any cache length is computed.
        max_seq_len = cast(int, self.max_seq_len)
        num_tokens = max(*self.image_prefill_chunk_sizes, cast(int, self.prefill_chunk_size))
        # `set_default_values` sets a partition length exactly when the attention is flash.
        partition_len = self.kvcache_partition_len
        if partition_len is not None:
            return max(align(max_seq_len + num_tokens, partition_len), 2 * partition_len)
        return align(max_seq_len + num_tokens, 64)

    @property
    def kvcache_block_size(self) -> int:
        # Eager attention keeps one block per prompt; flash attention one block per partition.
        return self.kvcache_partition_len or self.kvcache_len

    @property
    def kvcache_blocks_per_prompt(self) -> int:
        return self.kvcache_len // self.kvcache_block_size


__all__ = [
    "RBLNQwenImage21Transformer2DModelConfig",
]
