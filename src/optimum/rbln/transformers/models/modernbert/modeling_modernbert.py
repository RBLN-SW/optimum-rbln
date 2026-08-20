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

from typing import TYPE_CHECKING

import torch
from transformers.modeling_outputs import MaskedLMOutput

from ...modeling_generic import RBLNModelForMaskedLM
from .configuration_modernbert import RBLNModernBertForMaskedLMConfig
from .modernbert_architecture import ModernBertModelWrapper


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class RBLNModernBertForMaskedLM(RBLNModelForMaskedLM):
    """
    RBLN optimized ModernBERT model for masked language modeling tasks.

    This class provides hardware-accelerated inference for ModernBERT models
    on RBLN devices, supporting masked language modeling tasks such as
    token prediction and text completion.

    ModernBERT differs from the classic BERT / RoBERTa encoders in two ways
    that matter for compilation:

    - It alternates full and local (sliding-window) attention, building both 4D
      masks internally from the 2D padding mask. The compilation wrapper
      therefore forwards the padding mask unchanged instead of pre-expanding it
      (see `ModernBertModelWrapper`).
    - It selects its attention backend from `config._attn_implementation`. The
      model is loaded with `attn_implementation="sdpa"` so the compiler lowers
      attention through `scaled_dot_product_attention`; FlashAttention kernels
      are GPU-only and cannot be compiled for RBLN devices.
    """

    rbln_model_input_names = ["input_ids", "attention_mask"]

    @classmethod
    def get_pytorch_model(cls, *args, **kwargs) -> "PreTrainedModel":
        # Force the SDPA attention backend; FlashAttention-2 is CUDA-only and is
        # not compilable for RBLN. SDPA is numerically equivalent and is lowered
        # by the compiler to an RBLN attention kernel.
        kwargs["attn_implementation"] = "sdpa"
        return super().get_pytorch_model(*args, **kwargs)

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNModernBertForMaskedLMConfig
    ) -> torch.nn.Module:
        return ModernBertModelWrapper(model, rbln_config).eval()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | MaskedLMOutput:
        """
        Forward pass for the RBLN-optimized ModernBERT model for masked language modeling tasks.

        Args:
            input_ids (torch.LongTensor of shape (batch_size, sequence_length), optional): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional): Mask to avoid performing attention on padding token indices.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a MaskedLMOutput object.
        """
        return super().forward(input_ids, attention_mask, **kwargs)
