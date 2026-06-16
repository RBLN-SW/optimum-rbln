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

from typing import Any

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger


logger = get_logger()


class RBLNWhisperForConditionalGenerationConfig(RBLNModelConfig):
    """
    Configuration class for RBLNWhisperForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Whisper models for speech recognition and transcription tasks.
    """

    def __init__(
        self,
        batch_size: int = None,
        token_timestamps: bool = None,
        use_attention_mask: bool = None,
        enc_max_seq_len: int = None,
        dec_max_seq_len: int = None,
        kvcache_num_blocks: int = None,
        kvcache_block_size: int = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int, optional): The batch size for inference. Defaults to 1.
            token_timestamps (bool, optional): Whether to output token timestamps during generation. Defaults to False.
            use_attention_mask (bool, optional): Whether to use attention masks during inference. This is automatically
            enc_max_seq_len (int, optional): Maximum sequence length for the encoder.
            dec_max_seq_len (int, optional): Maximum sequence length for the decoder.
            kvcache_num_blocks (int, optional): The total number of blocks to allocate for the
                PagedAttention KV cache for the SelfAttention. Defaults to batch_size + 1.
            kvcache_block_size (int, optional): Sets the size (in number of tokens) of each block
                in the PagedAttention KV cache for the SelfAttention. Defaults to dec_max_seq_len.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)

        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.token_timestamps = token_timestamps or False
        self.enc_max_seq_len = enc_max_seq_len
        self.dec_max_seq_len = dec_max_seq_len

        self.use_attention_mask = use_attention_mask
        self.use_attention_mask = self.use_attention_mask or False
        self.kvcache_num_blocks = kvcache_num_blocks
        self.kvcache_block_size = kvcache_block_size
