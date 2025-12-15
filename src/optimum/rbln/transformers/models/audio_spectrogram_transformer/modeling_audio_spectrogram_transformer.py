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

from typing import TYPE_CHECKING, Optional

import torch
from transformers import AutoModelForAudioClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from .configuration_audio_spectrogram_transformer import RBLNASTForAudioClassificationConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, PretrainedConfig, PreTrainedModel


class RBLNASTForAudioClassification(RBLNModel):
    """
    Audio Spectrogram Transformer model with an audio classification head on top (a linear layer on top of the pooled output) e.g. for datasets like AudioSet, Speech Commands v2.
    This model inherits from [RBLNModelForAudioClassification]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformer-based ASTForAudioClassification models on RBLN devices.
    It implements the methods to convert a pre-trained transformers ASTForAudioClassification model into a RBLN transformer model by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN Compiler.
    """

    auto_model_class = AutoModelForAudioClassification

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: "AutoFeatureExtractor" = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: "PretrainedConfig" = None,
        rbln_config: Optional[RBLNASTForAudioClassificationConfig] = None,
    ) -> RBLNASTForAudioClassificationConfig:
        num_mel_bins = getattr(model_config, "num_mel_bins", None)

        if rbln_config.max_length is None:
            rbln_config.max_length = getattr(model_config, "max_length", None)
            for feature_extractor in preprocessors:
                if hasattr(feature_extractor, "max_length"):
                    rbln_config.max_length = feature_extractor.max_length
                    break

        if rbln_config.max_length is None:
            raise ValueError("max_length should be specified!")

        input_info = [
            (
                "input_values",
                [rbln_config.batch_size, rbln_config.max_length, num_mel_bins],
                "float32",
            ),
        ]

        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def forward(self, input_values: torch.Tensor, **kwargs) -> SequenceClassifierOutput:
        """
        Forward pass for the RBLN-optimized Audio Spectrogram Transformer model for audio classification.

        Args:
            input_values (torch.FloatTensor of shape (batch_size, max_length, num_mel_bins)):
                Float values mel features extracted from the raw audio waveform. Raw audio waveform can be obtained by
                loading a .flac or .wav audio file into an array of type list[float], a numpy.ndarray or a torch.Tensor, *e.g.* via
                the torchcodec library (pip install torchcodec) or the soundfile library (pip install soundfile).
                To prepare the array into input_features, the [AutoFeatureExtractor] should be used for extracting the
                mel features, padding and conversion into a tensor of type torch.FloatTensor.

        Returns:
            Returns a SequenceClassifierOutput object.
        """

        return super().forward(input_values, **kwargs)
