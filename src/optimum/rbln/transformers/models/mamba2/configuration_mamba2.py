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

from typing import List, Literal, Optional

from ....configuration_utils import RBLNModelConfig


PhaseType = Literal["prefill", "decode"]


class RBLNMamba2ForCausalLMConfig(RBLNModelConfig):
    """
    RBLN configuration for `Mamba2ForCausalLM`.

    Notes:
        - HuggingFace `Mamba2Config` does not define `max_position_embeddings`, so `max_seq_len`
          must be explicitly provided for compilation.
        - This config intentionally mirrors the high-level UX of `RBLNDecoderOnlyModelForCausalLMConfig`,
          but uses SSM cache tensors (conv/ssm states) instead of KV cache.
    """

    _default_phases: List[PhaseType] = ["prefill", "decode"]

    def __init__(
        self,
        batch_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        decoder_batch_sizes: Optional[List[int]] = None,
        phases: Optional[List[PhaseType]] = None,
        use_attention_mask: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = int(batch_size or 1)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        # Required for compilation (HF Mamba2Config doesn't provide a limit)
        self.max_seq_len = max_seq_len
        if self.max_seq_len is not None:
            self.max_seq_len = int(self.max_seq_len)
        if self.max_seq_len is not None and self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        
        self.prefill_chunk_size = 128
        if self.prefill_chunk_size % 64 != 0 or self.prefill_chunk_size <= 0:
            raise ValueError("`prefill_chunk_size` must be a positive integer divisible by 64.")

        self.phases = phases or list(self._default_phases)
        self.decoder_batch_sizes = None
        if "decode" in self.phases:
            self.decoder_batch_sizes = decoder_batch_sizes or [self.batch_size]
            if max(self.decoder_batch_sizes) > self.batch_size:
                raise ValueError(
                    f"Decoder batch size ({max(self.decoder_batch_sizes)}) must be <= batch_size ({self.batch_size})."
                )
            # Larger batch sizes first (matches decoderonly UX)
            self.decoder_batch_sizes.sort(reverse=True)

    @property
    def can_generate(self) -> bool:
        return "decode" in self.phases


