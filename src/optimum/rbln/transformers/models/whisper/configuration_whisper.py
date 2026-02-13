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

from __future__ import annotations

from pydantic import Field

from ....configuration_utils import PositiveIntDefaultOne, RBLNModelConfig
from ....utils.logging import get_logger


logger = get_logger()


class RBLNWhisperForConditionalGenerationConfig(RBLNModelConfig):
    """
    Configuration class for RBLNWhisperForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Whisper models for speech recognition and transcription tasks.
    """

    batch_size: PositiveIntDefaultOne = Field(default=1, description="The batch size for inference.")
    token_timestamps: bool = Field(default=False, description="Whether to output token timestamps during generation.")
    use_attention_mask: bool = Field(default=False, description="Whether to use attention masks during inference.")
    enc_max_seq_len: int | None = Field(default=None, description="Maximum sequence length for the encoder.")
    dec_max_seq_len: int | None = Field(default=None, description="Maximum sequence length for the decoder.")
    kvcache_num_blocks: int | None = Field(
        default=None,
        description="The total number of blocks to allocate for the PagedAttention KV cache "
        "for the SelfAttention. Defaults to batch_size.",
    )
    kvcache_block_size: int | None = Field(
        default=None,
        description="The size (in number of tokens) of each block in the PagedAttention KV cache "
        "for the SelfAttention. Defaults to dec_max_seq_len.",
    )
