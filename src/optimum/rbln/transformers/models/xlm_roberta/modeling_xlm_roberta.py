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

from typing import Optional, Union

import torch
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput

from ...modeling_generic import RBLNModelForSequenceClassification, RBLNTransformerEncoderForFeatureExtraction


class RBLNXLMRobertaModel(RBLNTransformerEncoderForFeatureExtraction):
    """
    XLM-RoBERTa base model optimized for RBLN NPU.
    """

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[BaseModelOutputWithPoolingAndCrossAttentions, tuple]:
        """
        Forward pass for the RBLN-optimized XLM-RoBERTa base model.

        Args:
            input_ids (torch.Tensor of shape (batch_size, sequence_length), optional): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor of shape (batch_size, sequence_length), optional): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor of shape (batch_size, sequence_length), optional): Segment token indices to indicate different portions of the inputs.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a BaseModelOutputWithPoolingAndCrossAttentions object.
        """

        if token_type_ids is not None:
            kwargs.setdefault("token_type_ids", token_type_ids)

        return super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


class RBLNXLMRobertaForSequenceClassification(RBLNModelForSequenceClassification):
    """
    XLM-RoBERTa model for sequence classification tasks optimized for RBLN NPU.
    """

    rbln_model_input_names = ["input_ids", "attention_mask"]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[SequenceClassifierOutput, tuple]:
        """
        Forward pass for the RBLN-optimized XLM-RoBERTa model for sequence classification.

        Args:
            input_ids (torch.LongTensor of shape (batch_size, sequence_length), optional): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.LongTensor of shape (batch_size, sequence_length), optional): Segment token indices to indicate first and second portions of the inputs.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a SequenceClassifierOutput object.
        """

        if token_type_ids is not None:
            kwargs.setdefault("token_type_ids", token_type_ids)

        return super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
