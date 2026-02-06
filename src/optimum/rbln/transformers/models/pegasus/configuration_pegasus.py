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

from ...configuration_generic import RBLNTransformerEncoderForFeatureExtractionConfig
from ..seq2seq import RBLNModelForSeq2SeqLMConfig


class RBLNPegasusModelConfig(RBLNTransformerEncoderForFeatureExtractionConfig):
    """
    Configuration class for RBLNPegasusModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized PEGASUS models for feature extraction tasks.
    """

    rbln_model_input_names = ["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"]


class RBLNPegasusForConditionalGenerationConfig(RBLNModelForSeq2SeqLMConfig):
    """
    Configuration class for RBLNPegasusForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized PEGASUS models for conditional text generation tasks.
    """

    support_paged_attention = True
