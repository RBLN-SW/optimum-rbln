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
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from ...modeling_generic import RBLNModelForQuestionAnswering


class RBLNDistilBertForQuestionAnswering(RBLNModelForQuestionAnswering):
    """
    RBLN optimized DistilBERT model for question answering tasks.

    This class provides hardware-accelerated inference for DistilBERT models
    on RBLN devices, supporting extractive question answering tasks where
    the model predicts start and end positions of answers in text.
    """

    rbln_model_input_names = ["input_ids", "attention_mask"]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        """
        Forward pass for the RBLN-optimized DistilBERT model for question answering tasks.

        Args:
            input_ids (torch.Tensor of shape (batch_size, sequence_length), optional): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor of shape (batch_size, sequence_length), optional): Mask to avoid performing attention on padding token indices.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a QuestionAnsweringModelOutput object.
        """

        return super().forward(input_ids, attention_mask, **kwargs)
