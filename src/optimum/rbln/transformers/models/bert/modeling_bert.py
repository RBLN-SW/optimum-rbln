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

from typing import Optional, Tuple, Union

import torch
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
)

from ...modeling_generic import (
    RBLNModelForMaskedLM,
    RBLNModelForQuestionAnswering,
    RBLNTransformerEncoderForFeatureExtraction,
)
from .bert_architecture import BertModelWrapper
from .configuration_bert import RBLNBertModelConfig


class RBLNBertModel(RBLNTransformerEncoderForFeatureExtraction):
    """
    RBLN optimized BERT model for feature extraction tasks.

    This class provides hardware-accelerated inference for BERT models
    on RBLN devices, optimized for extracting contextualized embeddings
    and features from text sequences.
    """

    rbln_model_input_names = ["input_ids", "attention_mask"]

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNBertModelConfig) -> torch.nn.Module:
        return BertModelWrapper(model, rbln_config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[BaseModelOutputWithPoolingAndCrossAttentions, Tuple]:
        """
        Forward pass for the RBLN-optimized BERT model for feature extraction tasks.

        Args:
            input_ids (torch.Tensor of shape (batch_size, sequence_length), optional): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor of shape (batch_size, sequence_length), optional): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor of shape (batch_size, sequence_length), optional): Segment token indices to indicate first and second portions of the inputs.
            position_ids (torch.Tensor of shape (batch_size, sequence_length), optional): Indices of positions of each input sequence tokens in the position embeddings.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a BaseModelOutputWithPoolingAndCrossAttentions object.
        """

        input_map = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
        }

        model_input_names = getattr(self.rbln_config, "model_input_names", None)
        if model_input_names is None:
            model_input_names = self.rbln_model_input_names

        ordered_inputs = [input_map[name] for name in model_input_names if name in input_map]

        return super().forward(*ordered_inputs, **kwargs)


class RBLNBertForMaskedLM(RBLNModelForMaskedLM):
    """
    RBLN optimized BERT model for masked language modeling tasks.

    This class provides hardware-accelerated inference for BERT models
    on RBLN devices, supporting masked language modeling tasks such as
    token prediction and text completion.
    """

    rbln_model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[MaskedLMOutput, Tuple]:
        """
        Forward pass for the RBLN-optimized BERT model for masked language modeling tasks.

        Args:
            input_ids (torch.Tensor of shape (batch_size, sequence_length), optional): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor of shape (batch_size, sequence_length), optional): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor of shape (batch_size, sequence_length), optional): Segment token indices to indicate first and second portions of the inputs.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a MaskedLMOutput object.
        """

        return super().forward(input_ids, attention_mask, token_type_ids, **kwargs)


class RBLNBertForQuestionAnswering(RBLNModelForQuestionAnswering):
    """
    RBLN optimized BERT model for question answering tasks.

    This class provides hardware-accelerated inference for BERT models
    on RBLN devices, supporting extractive question answering tasks where
    the model predicts start and end positions of answers in text.
    """

    rbln_model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[QuestionAnsweringModelOutput, Tuple]:
        """
        Forward pass for the RBLN-optimized BERT model for question answering tasks.

        Args:
            input_ids (torch.Tensor of shape (batch_size, sequence_length), optional): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor of shape (batch_size, sequence_length), optional): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.Tensor of shape (batch_size, sequence_length), optional): Segment token indices to indicate first and second portions of the inputs.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a QuestionAnsweringModelOutput object.
        """

        return super().forward(input_ids, attention_mask, token_type_ids, **kwargs)
