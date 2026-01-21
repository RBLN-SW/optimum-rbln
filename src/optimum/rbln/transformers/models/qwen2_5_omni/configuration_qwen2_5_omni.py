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


class RBLNQwen2_5OmniToken2WavDiTModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLNQwen2_5OmniToken2WavDiTModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Qwen2.5-Omni Token2Wav DiT model which takes speech tokens
    as input and predicts mel spectrogram.
    """

    def __init__(
        self,
        max_seq_len: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Args:
            max_seq_len (Optional[int]): Maximum sequence length for the DiT model.
                This determines the maximum number of mel frames that can be generated.
                The sequence length is calculated as `quantized_code.shape[1] * repeats`,
                where `repeats` is typically 6 (from config). If not provided, a default
                value will be used.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If `max_seq_len` is not a positive integer when provided.
        """
        super().__init__(**kwargs)

        if max_seq_len is not None and max_seq_len <= 0:
            raise ValueError(f"'max_seq_len' must be a positive integer, got {max_seq_len}.")

        self.max_seq_len = max_seq_len
