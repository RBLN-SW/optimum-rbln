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


class RBLNWav2Vec2ForCTCConfig(RBLNModelConfig):
    """
    Configuration class for RBLNWav2Vec2ForCTC.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Wav2Vec2 models for Connectionist Temporal Classification (CTC) tasks.
    """

    batch_size: PositiveIntDefaultOne = Field(default=1, description="The batch size for inference.")
    max_seq_len: int | None = Field(default=None, description="Maximum sequence length for the audio input.")
