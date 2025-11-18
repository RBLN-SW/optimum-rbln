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

from typing import Any, Optional

from ....configuration_utils import RBLNModelConfig
from ....utils.deprecation import deprecate_kwarg
from ....utils.logging import get_logger


logger = get_logger()


class RBLNModelForSeq2SeqLMConfig(RBLNModelConfig):
    support_paged_attention = None

    @deprecate_kwarg(old_name="pad_token_id", version="0.10.0")
    def __init__(
        self,
        batch_size: Optional[int] = None,
        enc_max_seq_len: Optional[int] = None,
        dec_max_seq_len: Optional[int] = None,
        use_attention_mask: Optional[bool] = None,
        kvcache_num_blocks: Optional[int] = None,
        kvcache_block_size: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            enc_max_seq_len (Optional[int]): Maximum sequence length for the encoder.
            dec_max_seq_len (Optional[int]): Maximum sequence length for the decoder.
            use_attention_mask (Optional[bool]): Whether to use attention masks during inference.
            kvcache_num_blocks (Optional[int]): The total number of blocks to allocate for the
                PagedAttention KV cache for the SelfAttention. Defaults to batch_size.
            kvcache_block_size (Optional[int]): Sets the size (in number of tokens) of each block
                in the PagedAttention KV cache for the SelfAttention. Defaults to dec_max_seq_len.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.enc_max_seq_len = enc_max_seq_len
        self.dec_max_seq_len = dec_max_seq_len

        self.use_attention_mask = use_attention_mask

        if self.support_paged_attention:
            self.kvcache_num_blocks = kvcache_num_blocks
            self.kvcache_block_size = kvcache_block_size
        else:
            if kvcache_num_blocks is not None or kvcache_block_size is not None:
                raise ValueError(
                    "You cannot set kvcache_num_blocks or kvcache_block_size as paged attention is not supported for the model."
                )
