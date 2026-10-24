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

from typing import Any

from ....configuration_utils import RBLNModelConfig


class RBLNFlux2KleinTextEncoderConfig(RBLNModelConfig):
    """Qwen3 hidden-state encoding for native FLUX.2 klein prompts."""

    def __init__(self, batch_size: int = 1, max_seq_len: int = 512, **kwargs: Any):
        """
        Args:
            batch_size (int): The compiled number of prompts.
            max_seq_len (int): The compiled prompt length, at most 512 tokens.
            kwargs: Additional arguments passed to RBLNModelConfig.
        """
        super().__init__(**kwargs)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(max_seq_len) is not int or not 1 <= max_seq_len <= 512:
            raise ValueError("max_seq_len must be between 1 and 512")
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len


__all__ = ["RBLNFlux2KleinTextEncoderConfig"]
