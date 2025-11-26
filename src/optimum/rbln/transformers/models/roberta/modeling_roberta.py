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

from typing import Tuple, Union

import torch
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput

from ...modeling_generic import RBLNModelForMaskedLM, RBLNModelForSequenceClassification


class RBLNRobertaForMaskedLM(RBLNModelForMaskedLM):
    """
    RBLN optimized RoBERTa model for masked language modeling tasks.

    This class provides hardware-accelerated inference for RoBERTa models
    on RBLN devices, supporting masked language modeling tasks such as
    token prediction and text completion.
    """

    rbln_model_input_names = ["input_ids", "attention_mask"]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Union[Tuple, MaskedLMOutput]:
        """
        Forward pass for the RBLN-optimized RoBERTa model for masked language modeling tasks.

        Args:
            input_ids (torch.LongTensor of shape (batch_size, sequence_length), optional): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional): Mask to avoid performing attention on padding token indices.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a MaskedLMOutput object.
        """
        return super().forward(input_ids, attention_mask, **kwargs)


class RBLNRobertaForSequenceClassification(RBLNModelForSequenceClassification):
    """
    RBLN optimized RoBERTa model for sequence classification tasks.

    This class provides hardware-accelerated inference for RoBERTa models
    on RBLN devices, supporting text classification tasks such as sentiment analysis,
    topic classification, and other sequence-level prediction tasks.
    """

    rbln_model_input_names = ["input_ids", "attention_mask"]

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        """
        Forward pass for the RBLN-optimized RoBERTa model for sequence classification tasks.

        Args:
            input_ids (torch.LongTensor of shape (batch_size, sequence_length), optional): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional): Mask to avoid performing attention on padding token indices.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a SequenceClassifierOutput object.
        """
        return super().forward(input_ids, attention_mask, **kwargs)
