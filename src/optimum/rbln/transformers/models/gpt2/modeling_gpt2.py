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

from ....utils import logging
from ...models.decoderonly import RBLNDecoderOnlyModel, RBLNDecoderOnlyModelForCausalLM
from .gpt2_architecture import GPT2Wrapper


logger = logging.get_logger(__name__)


class RBLNGPT2LMHeadModel(RBLNDecoderOnlyModelForCausalLM):
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).

    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the
    library implements for all its model.

    It implements the methods to convert a pre-trained transformers GPT2 model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    """

    _decoder_wrapper_cls = GPT2Wrapper
    _use_rotary_emb = False


class RBLNGPT2Model(RBLNDecoderOnlyModel):
    """
    The GPT2 Model transformer without a language modeling head.

    This model inherits from [`RBLNDecoderOnlyModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model.

    A class to convert and run pre-trained transformers based GPT2Model model on RBLN devices.
    It implements the methods to convert a pre-trained transformers GPT2Model model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.
    """

    _decoder_wrapper_cls = GPT2Wrapper
    _use_rotary_emb = False


__all__ = [
    "RBLNGPT2LMHeadModel",
    "RBLNGPT2Model",
]
