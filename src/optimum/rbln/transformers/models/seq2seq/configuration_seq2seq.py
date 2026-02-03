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

from typing import Any, ClassVar

from pydantic import field_validator

from ....configuration_utils import RBLNModelConfig
from ....utils.deprecation import deprecate_kwarg
from ....utils.logging import get_logger


logger = get_logger()


class RBLNModelForSeq2SeqLMConfig(RBLNModelConfig):
    """Configuration for RBLN Sequence-to-Sequence LM models."""

    support_paged_attention: ClassVar[bool | None] = None

    batch_size: int = 1
    enc_max_seq_len: int | None = None
    dec_max_seq_len: int | None = None
    use_attention_mask: bool | None = None
    kvcache_num_blocks: int | None = None
    kvcache_block_size: int | None = None

    @deprecate_kwarg(old_name="pad_token_id", version="0.10.0")
    def __init__(self, **data: Any):
        # Validate paged attention support
        kvcache_num_blocks = data.get("kvcache_num_blocks")
        kvcache_block_size = data.get("kvcache_block_size")

        if not self.support_paged_attention:
            if kvcache_num_blocks is not None or kvcache_block_size is not None:
                raise ValueError(
                    "You cannot set kvcache_num_blocks or kvcache_block_size as paged attention is not supported for the model."
                )
            # Remove these keys so they don't fail validation
            data.pop("kvcache_num_blocks", None)
            data.pop("kvcache_block_size", None)

        super().__init__(**data)

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        if v is None:
            return 1
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")
        return v
