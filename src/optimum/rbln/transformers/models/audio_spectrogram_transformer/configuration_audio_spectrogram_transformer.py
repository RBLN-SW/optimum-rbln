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

from pydantic import Field, field_validator

from ....configuration_utils import RBLNModelConfig


class RBLNASTForAudioClassificationConfig(RBLNModelConfig):
    """
    Configuration class for RBLNASTForAudioClassification.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Audio Spectrogram Transformer models for audio classification tasks.
    """

    batch_size: int = Field(default=1, description="The batch size for inference.")
    max_length: int | None = Field(
        default=None,
        description="Maximum length of the audio input in time dimension.",
    )

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        if v is None:
            return 1
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")
        return v
