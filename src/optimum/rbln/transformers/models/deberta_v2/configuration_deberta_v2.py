# Copyright 2026 Rebellions Inc. All rights reserved.

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

from ...configuration_generic import (
    RBLNModelForMaskedLMConfig,
    RBLNModelForQuestionAnsweringConfig,
    RBLNModelForSequenceClassificationConfig,
    RBLNTransformerEncoderForFeatureExtractionConfig,
)


class RBLNDebertaV2ModelConfig(RBLNTransformerEncoderForFeatureExtractionConfig):
    """Configuration for DeBERTa-v2 and DeBERTa-v3 feature extraction."""


class RBLNDebertaV2ForMaskedLMConfig(RBLNModelForMaskedLMConfig):
    """Configuration for DeBERTa-v2 and DeBERTa-v3 masked language modeling."""


class RBLNDebertaV2ForSequenceClassificationConfig(RBLNModelForSequenceClassificationConfig):
    """Configuration for DeBERTa-v2 and DeBERTa-v3 sequence classification."""


class RBLNDebertaV2ForQuestionAnsweringConfig(RBLNModelForQuestionAnsweringConfig):
    """Configuration for DeBERTa-v2 and DeBERTa-v3 question answering."""


__all__ = [
    "RBLNDebertaV2ModelConfig",
    "RBLNDebertaV2ForMaskedLMConfig",
    "RBLNDebertaV2ForSequenceClassificationConfig",
    "RBLNDebertaV2ForQuestionAnsweringConfig",
]
