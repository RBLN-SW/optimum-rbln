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


class RBLNWav2Vec2ForCTCConfig(RBLNModelConfig):
    """
    Configuration class for RBLNWav2Vec2ForCTC.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Wav2Vec2 models for Connectionist Temporal Classification (CTC) tasks.
    """

    def __init__(
        self,
        max_seq_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")
